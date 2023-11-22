"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

"""

import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
import sys
from utils import get_context_mask
from panopticapi.utils import rgb2id
import json
from lvis import LVIS
import pycocotools.mask as mask_util
import glob

# hyper-params
file_split = int(sys.argv[1])
total_split = int(sys.argv[2])
checkpoint_path = str(sys.argv[3])
save_fold_path = str(sys.argv[4])
if not os.path.exists(save_fold_path):
    os.makedirs(save_fold_path)
INPUT_SIZE = 1120
CONTEXT_ENLARGE_RATIO = 0.5
# The float16 affects final results slightly. Results in paper are obtained with float32.
TEST_DTYPE = torch.float16

# prepare model and instructions, which will be shared across all test cases
from modeling.factory import create_model_and_transforms
osm, processor = create_model_and_transforms()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
msd = checkpoint["model_state_dict"]
msd = {k.replace("module.", ""): v for k, v in msd.items()}
osm.load_state_dict(msd, False)
osm.to(dtype=TEST_DTYPE, device="cuda")
processor.image_processor.size = {"height": osm.input_size, "width": osm.input_size}
processor.tokenizer.padding_side = "left"
processor.qformer_tokenizer.padding_side = "left"
input_text = "What is in the segmentation mask? Assistant:"
lang_x = processor.tokenizer(
    [input_text],
    return_tensors="pt",
)
qformer_lang_x = processor.qformer_tokenizer(
    [input_text],
    return_tensors="pt",
)

### help function
def preprocess_image(pil_img):
    if min(pil_img.size) == max(pil_img.size):
        image = T.functional.resize(pil_img, size=INPUT_SIZE, interpolation=T.functional.InterpolationMode.BICUBIC)
    else:
        image = T.functional.resize(pil_img, size=INPUT_SIZE-1, max_size=INPUT_SIZE, interpolation=T.functional.InterpolationMode.BICUBIC)
    image = np.array(image)
    padded_image = np.zeros(shape=(INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)   
    padded_image[:image.shape[0], :image.shape[1]] = image
    image = Image.fromarray(padded_image)
    image = processor(images=image, return_tensors="pt")["pixel_values"].view(1, 3, INPUT_SIZE, INPUT_SIZE)
    return image

def preprocess_mask_coco(pil_mask, ignore_value=-1):
    if min(pil_mask.size) == max(pil_mask.size):
        mask = T.functional.resize(pil_mask, size=INPUT_SIZE, interpolation=T.functional.InterpolationMode.NEAREST)
    else:
        mask = T.functional.resize(pil_mask, size=INPUT_SIZE-1, max_size=INPUT_SIZE, interpolation=T.functional.InterpolationMode.NEAREST)
    mask = rgb2id(np.array(mask))
    assert (mask == ignore_value).sum() == 0
    padded_mask = np.ones(shape=(INPUT_SIZE, INPUT_SIZE), dtype=np.int32) * ignore_value
    padded_mask[:mask.shape[0], :mask.shape[1]] = mask
    return padded_mask

def preprocess_mask_lvis(pil_mask):
    if min(pil_mask.size) == max(pil_mask.size):
        mask = T.functional.resize(pil_mask, size=INPUT_SIZE, interpolation=T.functional.InterpolationMode.NEAREST)
    else:
        mask = T.functional.resize(pil_mask, size=INPUT_SIZE-1, max_size=INPUT_SIZE, interpolation=T.functional.InterpolationMode.NEAREST)
    mask = np.array(mask)
    padded_mask = np.zeros(shape=(INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    padded_mask[:mask.shape[0], :mask.shape[1]] = mask
    return padded_mask

def preprocess_mask_a847(pil_mask, ignore_value=65535):
    if min(pil_mask.size) == max(pil_mask.size):
        mask = T.functional.resize(pil_mask, size=INPUT_SIZE, interpolation=T.functional.InterpolationMode.NEAREST)
    else:
        mask = T.functional.resize(pil_mask, size=INPUT_SIZE-1, max_size=INPUT_SIZE, interpolation=T.functional.InterpolationMode.NEAREST)
    mask = np.array(mask)
    padded_mask = np.ones(shape=(INPUT_SIZE, INPUT_SIZE), dtype=np.int32) * ignore_value
    padded_mask[:mask.shape[0], :mask.shape[1]] = mask
    return padded_mask

def get_pred_class_and_prob(image, binary_mask, cache_flag, dataset_type):
    context_mask = get_context_mask(binary_mask, INPUT_SIZE, CONTEXT_ENLARGE_RATIO).view(1, 1, INPUT_SIZE, INPUT_SIZE)
    generated_output = osm.generate(
        pixel_values=image.cuda().to(TEST_DTYPE),
        qformer_input_ids=qformer_lang_x["input_ids"].cuda(),
        qformer_attention_mask=qformer_lang_x["attention_mask"].cuda(),
        input_ids=lang_x["input_ids"].cuda(),
        attention_mask=lang_x["attention_mask"].cuda(),
        cache_image_embeds=cache_flag,
        segmentation_mask=binary_mask.cuda(),
        input_context_mask=context_mask.cuda(),
        # 12 could be too much considering most gt are single word
        max_new_tokens=12,
        num_beams=1,
        dataset_type=dataset_type,
        return_dict_in_generate=True,
        output_scores=True
    )
    generated_text = generated_output["sequences"][0]
    scores = generated_output["scores"]
    pred_class = processor.tokenizer.decode(generated_text).split('</s>')[1].strip()
    pred_class_tokenidx = processor.tokenizer.encode(pred_class)
    scores = scores[:len(pred_class_tokenidx) -1] # minus one for bos token
    # matching the pred_class_tokenidx for prob computation
    temp = 1.0
    probs = [torch.nn.functional.softmax(score / temp, dim=-1) for score in scores]
    pred_prob = 1.0
    for p_idx, prob in enumerate(probs):
        pred_idx = pred_class_tokenidx[p_idx + 1]
        pred_prob *= prob[0, pred_idx].cpu().item()
    #print(f"pred_class: {pred_class}, pred_prob: {pred_prob}")
    return pred_class, pred_prob

### Testing functions
# COCO
def test_coco(coco_root, json_file_path):
    json_file = json.load(open(json_file_path))["annotations"]
    num_split = len(json_file) // total_split
    if file_split != total_split - 1:
        json_file = json_file[file_split * num_split: (file_split+1) * num_split]
    else:
        json_file = json_file[file_split * num_split:]

    for idx in range(len(json_file)):
        test_case = json_file[idx]
        file_name = test_case["file_name"].replace(".jpg", "").replace(".png", "")
        image = Image.open(f"{coco_root}/val2017/{file_name}.jpg").convert("RGB")
        image = preprocess_image(image)
        pan_seg = Image.open(f"{coco_root}/panoptic_val2017/{file_name}.png").convert("RGB")
        pan_seg = preprocess_mask_coco(pan_seg)
        cache_flag = True
        for s_idx in range(len(test_case["segments_info"])):
            segment_info = test_case["segments_info"][s_idx]
            if segment_info.get("iscrowd", 0):
                continue
            binary_mask = (pan_seg == segment_info["id"])
            if binary_mask.sum() < 1:
                continue
            binary_mask = torch.from_numpy(np.ascontiguousarray(binary_mask.copy().reshape(1, 1, INPUT_SIZE, INPUT_SIZE))).float()
            pred_class_and_prob = {}
            for dataset_type in ["coco", "any"]:
                pred_class, pred_prob = get_pred_class_and_prob(
                    image, binary_mask, cache_flag, dataset_type)
                cache_flag = False
                pred_class_and_prob[dataset_type] = {"class": pred_class, "prob": pred_prob}
            segment_info["open_end_pred"] = pred_class_and_prob
            test_case["segments_info"][s_idx] = segment_info
        json_file[idx] = test_case
    
    save_path = f"{save_fold_path}/coco_pred_{file_split}_of_{total_split}.json"
    print(f"Saving prediction results at {save_path}")
    with open(save_path, 'w') as f:
        json.dump(json_file, f)
    return

# ADE20k
def test_ade20k(ade20k_root, json_file_path):
    json_file = json.load(open(json_file_path))["annotations"]
    num_split = len(json_file) // total_split
    if file_split != total_split - 1:
        json_file = json_file[file_split * num_split: (file_split+1) * num_split]
    else:
        json_file = json_file[file_split * num_split:]

    for idx in range(len(json_file)):
        test_case = json_file[idx]
        file_name = test_case["file_name"].replace(".jpg", "").replace(".png", "")
        image = Image.open(f"{ade20k_root}/images/validation/{file_name}.jpg").convert("RGB")
        image = preprocess_image(image)
        pan_seg = Image.open(f"{ade20k_root}/ade20k_panoptic_val/{file_name}.png").convert("RGB")
        pan_seg = preprocess_mask_coco(pan_seg)
        cache_flag = True
        for s_idx in range(len(test_case["segments_info"])):
            segment_info = test_case["segments_info"][s_idx]
            if segment_info.get("iscrowd", 0):
                continue
            binary_mask = (pan_seg == segment_info["id"])
            if binary_mask.sum() < 1:
                continue
            binary_mask = torch.from_numpy(np.ascontiguousarray(binary_mask.copy().reshape(1, 1, INPUT_SIZE, INPUT_SIZE))).float()
            pred_class_and_prob = {}
            for dataset_type in ["ade20k", "any"]:
                pred_class, pred_prob = get_pred_class_and_prob(
                    image, binary_mask, cache_flag, dataset_type)
                cache_flag = False
                pred_class_and_prob[dataset_type] = {"class": pred_class, "prob": pred_prob}
            segment_info["open_end_pred"] = pred_class_and_prob
            test_case["segments_info"][s_idx] = segment_info
        json_file[idx] = test_case
    
    save_path = f"{save_fold_path}/ade20k_pred_{file_split}_of_{total_split}.json"
    print(f"Saving prediction results at {save_path}")
    with open(save_path, 'w') as f:
        json.dump(json_file, f)
    return

# Citysapes Panoptic
def test_cityscapes(cityscapes_root, json_file_path):
    json_file = json.load(open(json_file_path))["annotations"]
    num_split = len(json_file) // total_split
    if file_split != total_split - 1:
        json_file = json_file[file_split * num_split: (file_split+1) * num_split]
    else:
        json_file = json_file[file_split * num_split:]

    for idx in range(len(json_file)):
        test_case = json_file[idx]
        # e.g. frankfurt_000000_000294_gtFine_panoptic.png
        file_name = test_case["file_name"].replace(".jpg", "").replace(".png", "")
        img_name = file_name.replace("_gtFine_panoptic", "_leftImg8bit")
        image = Image.open(f"{cityscapes_root}/leftImg8bit/val/{img_name.split('_')[0]}/{img_name}.png").convert("RGB")
        image = preprocess_image(image)
        pan_seg = Image.open(f"{cityscapes_root}/gtFine/cityscapes_panoptic_val/{file_name}.png").convert("RGB")
        pan_seg = preprocess_mask_coco(pan_seg)
        cache_flag = True
        for s_idx in range(len(test_case["segments_info"])):
            segment_info = test_case["segments_info"][s_idx]
            if segment_info.get("iscrowd", 0):
                continue
            binary_mask = (pan_seg == segment_info["id"])
            if binary_mask.sum() < 1:
                continue
            binary_mask = torch.from_numpy(np.ascontiguousarray(binary_mask.copy().reshape(1, 1, INPUT_SIZE, INPUT_SIZE))).float()
            pred_class_and_prob = {}
            for dataset_type in ["cityscapes", "any"]:
                pred_class, pred_prob = get_pred_class_and_prob(
                    image, binary_mask, cache_flag, dataset_type)
                cache_flag = False
                pred_class_and_prob[dataset_type] = {"class": pred_class, "prob": pred_prob}
            segment_info["open_end_pred"] = pred_class_and_prob
            test_case["segments_info"][s_idx] = segment_info
        json_file[idx] = test_case
    
    save_path = f"{save_fold_path}/cityscapes_pred_{file_split}_of_{total_split}.json"
    print(f"Saving prediction results at {save_path}")
    with open(save_path, 'w') as f:
        json.dump(json_file, f)
    return

# LVIS
def test_lvis(lvis_root, json_file_path):
    lvis_api = LVIS(json_file_path)
    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file_path
    )
    imgs_anns = list(zip(imgs, anns))
    print("Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file_path))
    def get_file_name(img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(img_root + '/' + split_folder, file_name)

    num_split = len(imgs_anns) // total_split
    if file_split != total_split - 1:
        imgs_anns = imgs_anns[file_split * num_split: (file_split+1) * num_split]
    else:
        imgs_anns = imgs_anns[file_split * num_split:]

    for idx in range(len(imgs_anns)):
        img_dict, anno_dict_list = imgs_anns[idx]
        file_name = get_file_name(lvis_root, img_dict)
        image = Image.open(file_name).convert("RGB")
        image = preprocess_image(image)
        cache_flag = True
        for s_idx in range(len(anno_dict_list)):
            anno = anno_dict_list[s_idx]
            assert anno["image_id"] == img_dict["id"]
            if anno.get("iscrowd", 0):
                continue
            segm = anno["segmentation"]
            valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            assert len(segm) == len(
                valid_segm
            ), "Annotation contains an invalid polygon with < 3 points"
            assert len(segm) > 0
            rles = mask_util.frPyObjects(segm, img_dict["height"], img_dict["width"])
            rle = mask_util.merge(rles)
            binary_mask = mask_util.decode(rle).astype(bool)
            binary_mask = Image.fromarray(binary_mask.astype(np.uint8))
            binary_mask = preprocess_mask_lvis(binary_mask)
            if binary_mask.sum() < 1:
                continue
            binary_mask = torch.from_numpy(np.ascontiguousarray(binary_mask.copy().reshape(1, 1, INPUT_SIZE, INPUT_SIZE))).float()
            pred_class_and_prob = {}
            for dataset_type in ["lvis", "any"]:
                pred_class, pred_prob = get_pred_class_and_prob(
                    image, binary_mask, cache_flag, dataset_type)
                cache_flag = False
                pred_class_and_prob[dataset_type] = {"class": pred_class, "prob": pred_prob}
            anno["open_end_pred"] = pred_class_and_prob
            anno_dict_list[s_idx] = anno
        imgs_anns[idx] = [img_dict, anno_dict_list]
    
    save_path = f"{save_fold_path}/lvis_pred_{file_split}_of_{total_split}.json"
    print(f"Saving prediction results at {save_path}")
    with open(save_path, 'w') as f:
        json.dump({"pred": imgs_anns}, f)
    return

# A-847
def test_a847(a847_root):
    file_list = glob.glob(f"{a847_root}/images_detectron2/validation/*.jpg")
    num_split = len(file_list) // total_split
    if file_split != total_split - 1:
        file_list = file_list[file_split * num_split: (file_split+1) * num_split]
    else:
        file_list = file_list[file_split * num_split:]
    result_list = []
    for idx in range(len(file_list)):
        result_dict = {}
        file_name = file_list[idx]
        result_dict["file_name"] = file_name
        seg_file = file_name.replace("/images_detectron2/", "/annotations_detectron2/").replace(".jpg", ".tif")
        image = Image.open(file_name).convert("RGB")
        image = preprocess_image(image)
        seg = Image.open(seg_file) # tiff
        seg = preprocess_mask_a847(seg)
        cache_flag = True
        all_ids = np.unique(seg)
        result_dict["pred"] = []
        for unique_id in all_ids:
            if unique_id == 65535:
                continue
            binary_mask = (seg == unique_id)
            if binary_mask.sum() < 1:
                continue
            binary_mask = torch.from_numpy(np.ascontiguousarray(binary_mask.copy().reshape(1, 1, INPUT_SIZE, INPUT_SIZE))).float()
            pred_class_and_prob = {}
            for dataset_type in ["a847", "any"]:
                pred_class, pred_prob = get_pred_class_and_prob(
                    image, binary_mask, cache_flag, dataset_type)
                cache_flag = False
                pred_class_and_prob[dataset_type] = {"class": pred_class, "prob": pred_prob}
            result_dict["pred"].append({
                "mask_unique_id": int(unique_id),
                "open_end_pred": pred_class_and_prob
            })

        result_list.append(result_dict)
    
    save_path = f"{save_fold_path}/a847_pred_{file_split}_of_{total_split}.json"
    print(f"Saving prediction results at {save_path}")
    with open(save_path, 'w') as f:
        json.dump({"pred": result_list}, f)
    return

# PC-459
def test_pc459(pc459_root):
    file_list = glob.glob(f"{pc459_root}/images/validation/*.jpg")
    num_split = len(file_list) // total_split
    if file_split != total_split - 1:
        file_list = file_list[file_split * num_split: (file_split+1) * num_split]
    else:
        file_list = file_list[file_split * num_split:]
    result_list = []
    for idx in range(len(file_list)):
        result_dict = {}
        file_name = file_list[idx]
        result_dict["file_name"] = file_name
        seg_file = file_name.replace("/images/", "/annotations_ctx459/").replace(".jpg", ".tif")
        image = Image.open(file_name).convert("RGB")
        image = preprocess_image(image)
        seg = Image.open(seg_file) # tiff
        seg = preprocess_mask_a847(seg) # shared by a847 and pc459
        cache_flag = True
        all_ids = np.unique(seg)
        result_dict["pred"] = []
        for unique_id in all_ids:
            # 430 is for "unknown"
            if unique_id in [65535, 430]:
                continue
            binary_mask = (seg == unique_id)
            if binary_mask.sum() < 1:
                continue
            binary_mask = torch.from_numpy(np.ascontiguousarray(binary_mask.copy().reshape(1, 1, INPUT_SIZE, INPUT_SIZE))).float()
            pred_class_and_prob = {}
            for dataset_type in ["pc459", "any"]:
                pred_class, pred_prob = get_pred_class_and_prob(
                    image, binary_mask, cache_flag, dataset_type)
                cache_flag = False
                pred_class_and_prob[dataset_type] = {"class": pred_class, "prob": pred_prob}
            result_dict["pred"].append({
                "mask_unique_id": int(unique_id),
                "open_end_pred": pred_class_and_prob
            })

        result_list.append(result_dict)
    
    save_path = f"{save_fold_path}/pc459_pred_{file_split}_of_{total_split}.json"
    print(f"Saving prediction results at {save_path}")
    with open(save_path, 'w') as f:
        json.dump({"pred": result_list}, f)
    return


if __name__ == "__main__":
    # Update the dataset root accordingly
    coco_root = "coco"
    coco_json_file_path = f"{coco_root}/annotations/panoptic_val2017.json"
    ade20k_root = "ADEChallengeData2016"
    ade20k_json_file_path = f"{ade20k_root}/ade20k_panoptic_val.json"
    cityscapes_root = "cityscapes"
    cityscapes_json_file_path = f"{cityscapes_root}/gtFine/cityscapes_panoptic_val.json"
    lvis_root = "lvis"
    lvis_json_file_path = f"{lvis_root}/lvis_v1_val.json"
    a847_root = "ADE20K_2021_17_01"
    pc459_root = "pascal_ctx_d2"
    
    print("Testing COCO Panoptic...")
    test_coco(coco_root, coco_json_file_path)
    print("Testing ADE20K Panoptic...")
    test_ade20k(ade20k_root, ade20k_json_file_path)
    print("Testing Cityscapes Panoptic...")
    test_cityscapes(cityscapes_root, cityscapes_json_file_path)
    print("Testing LVIS Instance...")
    test_lvis(lvis_root, lvis_json_file_path)
    print("Testing A847 Semantic...")
    test_a847(a847_root)
    print("Testing PC459 Semantic...")
    test_pc459(pc459_root)