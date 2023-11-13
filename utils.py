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

from modeling.factory import create_model_and_transforms
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def prepare_sam(
    sam_checkpoint="sam_vit_h_4b8939.pth",
    model_type="vit_h",
    device="cuda",
    points_per_side=32,
    pred_iou_thresh=0.95,
    stability_score_thresh=0.95,
    min_mask_region_area=800,
):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    # frist for fully automatic seg/cls, second for interactive one.
    return mask_generator, SamPredictor(sam)


def prepare_osm(
    osm_checkpoint="osm.pth",
    device="cuda"
):
    osm, processor = create_model_and_transforms()
    checkpoint = torch.load(osm_checkpoint, map_location="cpu")
    msd = checkpoint["model_state_dict"]
    msd = {k.replace("module.", ""): v for k, v in msd.items()}
    osm.load_state_dict(msd, False)
    osm.to(dtype=torch.float16, device=device)

    processor.image_processor.size = {"height": osm.input_size, "width": osm.input_size}
    processor.tokenizer.padding_side = "left"
    processor.qformer_tokenizer.padding_side = "left"
    return osm, processor


def prepare_instruction(
    processor,
    input_text = "What is in the segmentation mask? Assistant:"     
):    
    lang_x = processor.tokenizer(
        [input_text],
        return_tensors="pt",
    )
    qformer_lang_x = processor.qformer_tokenizer(
        [input_text],
        return_tensors="pt",
    )
    return lang_x, qformer_lang_x


def prepare_image(
    image_path,
    input_size
):
    image = Image.open(image_path).convert("RGB")
    if min(image.size) == max(image.size):
        image = T.functional.resize(image, size=input_size, interpolation=T.functional.InterpolationMode.BICUBIC)
    else:
        image = T.functional.resize(image, size=input_size-1, max_size=input_size, interpolation=T.functional.InterpolationMode.BICUBIC)

    image_for_seg = np.array(image)

    # pad to input_size x input_size
    padded_image = np.zeros(shape=(input_size, input_size, 3), dtype=np.uint8)
    padded_image[:image_for_seg.shape[0], :image_for_seg.shape[1]] = image_for_seg
    image_for_osm = Image.fromarray(padded_image)
    return image_for_osm, image_for_seg


def get_masks(image_for_seg, mask_generator, input_size):
    masks = mask_generator.generate(image_for_seg)
    def process_mask(m):
        m = Image.fromarray(m)
        if min(m.size) == max(m.size):
            m = T.functional.resize(m, size=input_size, interpolation=T.functional.InterpolationMode.NEAREST)
        else:
            m = T.functional.resize(m, size=input_size-1,
                                    max_size=input_size, interpolation=T.functional.InterpolationMode.NEAREST)
        m = np.array(m)
        return m

    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    sorted_masks = [process_mask(m["segmentation"]) for m in sorted_masks] # each one is a H x W binary mask  
    print(f"Obtained {len(sorted_masks)} masks from mask generator")
    return sorted_masks


def get_context_mask(mask, input_size, enlarge_ratio=0.5):
    if mask.sum() == 0:
        raise ValueError("Got an empty mask!")
    
    if enlarge_ratio < 0:
        return torch.ones_like(mask).view(1, input_size, input_size)

    mask = mask.view(input_size, input_size)
    rows, cols = torch.where(mask)
    min_row, min_col = rows.min().item(), cols.min().item()
    max_row, max_col = rows.max().item(), cols.max().item()

    row_size = max_row - min_row + 1
    col_size = max_col - min_col + 1
    min_row = max(0, int(min_row - row_size * enlarge_ratio))
    max_row = min(input_size-1, int(max_row + row_size * enlarge_ratio))
    min_col = max(0, int(min_col - col_size * enlarge_ratio))
    max_col = min(input_size-1, int(max_col + col_size * enlarge_ratio))
    context_mask = torch.zeros_like(mask)
    context_mask[min_row:max_row+1, min_col:max_col+1] = 1
    return context_mask.view(1, input_size, input_size)


def get_classes(image, masks, class_generator, processor, 
                qformer_lang_x, lang_x):
    input_size = processor.image_processor.size["height"]
    image = processor(images=image, return_tensors="pt")["pixel_values"].view(1, 3, input_size, input_size)

    classes = []
    class_probs = []

    for binary_mask in masks:
        # padding
        padded_binary_mask = np.zeros(shape=(input_size, input_size), dtype=np.uint8)
        padded_binary_mask[:binary_mask.shape[0], :binary_mask.shape[1]] = binary_mask
        binary_mask = padded_binary_mask
        binary_mask = torch.from_numpy(np.ascontiguousarray(binary_mask.copy().reshape(1, input_size, input_size)))

        if binary_mask.sum() < 100:
            classes.append("")
            class_probs.append(0)
            continue

        binary_mask = binary_mask.view(1, 1, input_size, input_size).float()
        context_mask = get_context_mask(binary_mask, input_size, 0.5).view(1, 1, input_size, input_size)

        generated_output = class_generator.generate(
            pixel_values=image.cuda().to(torch.float16),
            qformer_input_ids=qformer_lang_x["input_ids"].cuda(),
            qformer_attention_mask=qformer_lang_x["attention_mask"].cuda(),
            input_ids=lang_x["input_ids"].cuda(),
            attention_mask=lang_x["attention_mask"].cuda(),
            cache_image_embeds=(len(classes) == 0),
            segmentation_mask=binary_mask.cuda(),
            input_context_mask=context_mask.cuda(),
            dataset_type="any",
            max_new_tokens=16,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True)
        
        generated_text = generated_output["sequences"][0]
        scores = generated_output["scores"]

        # get pred_class
        pred_class = processor.tokenizer.decode(generated_text).split('</s>')[1].strip()

        # get pred_class probability
        pred_class_tokenidx = processor.tokenizer.encode(pred_class)
        scores = scores[:len(pred_class_tokenidx) -1] # minus one for bos token
        temp = 1.0
        probs = [torch.nn.functional.softmax(score / temp, dim=-1) for score in scores]
        pred_prob = 1.0
        for p_idx, prob in enumerate(probs):
            pred_idx = pred_class_tokenidx[p_idx + 1]
            pred_prob *= prob[0, pred_idx].cpu().numpy()

        classes.append(pred_class)
        class_probs.append(pred_prob)

    print("predcited class names:", classes)
    print("predcited class probs:", class_probs)

    return classes, class_probs
