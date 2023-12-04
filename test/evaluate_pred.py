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

import json
from dataset_names import openseg_classes

# helper
def augment_gt_classes_with_plural_and_singular(gt_classes):
    singular = []
    plural = []
    for gt_class in gt_classes:
        plural.append(gt_class + 's')
        plural.append(gt_class + 'es')
        if gt_class.endswith("y"):
            plural.append(gt_class[:-1] + 'ies')
        if gt_class.endswith("f"):
            plural.append(gt_class[:-1] + 'ves')
        if gt_class.endswith("fe"):
            plural.append(gt_class[:-2] + 'ves')

        if gt_class.endswith("s"):
            singular.append(gt_class[:-1])
        if gt_class.endswith("es"):
            singular.append(gt_class[:-2])
        if gt_class.endswith("ies"):
            singular.append(gt_class[:-3] + 'y')
        if gt_class.endswith("ves"):
            singular.append(gt_class[:-3] + 'f')
            singular.append(gt_class[:-3] + 'fe')

    gt_classes = gt_classes + singular + plural
    # Note this does not affect our prediction, just for enlarging our gt set for a better matching and evaluation.
    return gt_classes


# COCO Panoptic
def get_all_coco_class():
    coco_meta = openseg_classes.get_coco_categories_with_prompt_eng()
    coco_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": x["isthing"]} for x in coco_meta}
    return coco_class
all_coco_class = get_all_coco_class()
def coco_id2class(idx):
    return all_coco_class[idx]["name"]
all_coco_vocabularies = []
for x in all_coco_class:
    all_coco_vocabularies += all_coco_class[x]["name"]
# convert to dict to speed-up
all_coco_vocabularies = {x:1 for x in all_coco_vocabularies}
def parse_coco_panoptic(json_file_list):
    total_sample = 0
    corr_sample_any = 0
    corr_sample_coco = 0
    not_in_vocab_any = 0
    not_in_vocab_coco = 0
    for json_file in json_file_list:
        json_log = json.load(open(json_file))
        for img_log in json_log:
            segments_info = img_log["segments_info"]
            for segment_info in segments_info:
                if segment_info.get("iscrowd", 0):
                    continue
                gt_classes = coco_id2class(segment_info["category_id"])
                if not isinstance(gt_classes, list) or isinstance(gt_classes, tuple):
                    gt_classes = [gt_classes]
                gt_classes = augment_gt_classes_with_plural_and_singular(gt_classes)
                pred_class = segment_info["open_end_pred"]
                pred_class_coco = pred_class["coco"]["class"].lower()
                pred_class_any = pred_class["any"]["class"].lower()

                total_sample += 1
                corr_sample_any += int(pred_class_any in gt_classes)
                corr_sample_coco += int(pred_class_coco in gt_classes)
                not_in_vocab_any += int(pred_class_any not in all_coco_vocabularies)
                not_in_vocab_coco += int(pred_class_coco not in all_coco_vocabularies)

    print(f"\nCOCO Panoptic evaluation:")
    print(f"Agnostic: Acc: {corr_sample_any}/{total_sample} = {corr_sample_any/total_sample}, Not-In-Vocab: {not_in_vocab_any}/{total_sample} = {not_in_vocab_any/total_sample}")
    print(f"Specific: Acc: {corr_sample_coco}/{total_sample} = {corr_sample_coco/total_sample}, Not-In-Vocab: {not_in_vocab_coco}/{total_sample} = {not_in_vocab_coco/total_sample}")


# ADE20K Panoptic
def get_all_ade20k_class():
    ade20k_meta = openseg_classes.get_ade20k_categories_with_prompt_eng()
    ade20k_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": x["isthing"]} for x in ade20k_meta}
    return ade20k_class
all_ade20k_class = get_all_ade20k_class()
def ade20k_id2class(idx):
    return all_ade20k_class[idx]["name"]
all_ade20k_vocabularies = []
for x in all_ade20k_class:
    all_ade20k_vocabularies += all_ade20k_class[x]["name"]
# convert to dict to speed-up
all_ade20k_vocabularies = {x:1 for x in all_ade20k_vocabularies}
def parse_ade20k_panoptic(json_file_list):
    total_sample = 0
    corr_sample_any = 0
    corr_sample_ade20k = 0
    not_in_vocab_any = 0
    not_in_vocab_ade20k = 0
    for json_file in json_file_list:
        json_log = json.load(open(json_file))
        for img_log in json_log:
            segments_info = img_log["segments_info"]
            for segment_info in segments_info:
                if segment_info.get("iscrowd", 0):
                    continue
                gt_classes = ade20k_id2class(segment_info["category_id"])
                if not isinstance(gt_classes, list) or isinstance(gt_classes, tuple):
                    gt_classes = [gt_classes]
                gt_classes = augment_gt_classes_with_plural_and_singular(gt_classes)
                pred_class = segment_info["open_end_pred"]
                pred_class_ade20k = pred_class["ade20k"]["class"].lower()
                pred_class_any = pred_class["any"]["class"].lower()

                total_sample += 1
                corr_sample_any += int(pred_class_any in gt_classes)
                corr_sample_ade20k += int(pred_class_ade20k in gt_classes)
                not_in_vocab_any += int(pred_class_any not in all_ade20k_vocabularies)
                not_in_vocab_ade20k += int(pred_class_ade20k not in all_ade20k_vocabularies)


    print(f"\nADE20K Panoptic evaluation:")
    print(f"Agnostic: Acc: {corr_sample_any}/{total_sample} = {corr_sample_any/total_sample}, Not-In-Vocab: {not_in_vocab_any}/{total_sample} = {not_in_vocab_any/total_sample}")
    print(f"Specific: Acc: {corr_sample_ade20k}/{total_sample} = {corr_sample_ade20k/total_sample}, Not-In-Vocab: {not_in_vocab_ade20k}/{total_sample} = {not_in_vocab_ade20k/total_sample}")

# Cityscapes Panoptic
def get_all_cityscapes_class():
    cityscapes_meta = openseg_classes.get_cityscapes_categories_with_prompt_eng()
    cityscapes_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": x["isthing"]} for x in cityscapes_meta}
    return cityscapes_class

all_cityscapes_class = get_all_cityscapes_class()
def cityscapes_id2class(idx):
    return all_cityscapes_class[idx]["name"]
all_cityscapes_vocabularies = []
for x in all_cityscapes_class:
    all_cityscapes_vocabularies += all_cityscapes_class[x]["name"]
# convert to dict to speed-up
all_cityscapes_vocabularies = {x:1 for x in all_cityscapes_vocabularies}
def parse_cityscapes_panoptic(json_file_list):
    total_sample = 0
    corr_sample_any = 0
    corr_sample_cityscapes = 0
    not_in_vocab_any = 0
    not_in_vocab_cityscapes = 0
    for json_file in json_file_list:
        json_log = json.load(open(json_file))
        for img_log in json_log:
            segments_info = img_log["segments_info"]
            for segment_info in segments_info:
                if segment_info.get("iscrowd", 0):
                    continue
                gt_classes = cityscapes_id2class(segment_info["category_id"])
                if not isinstance(gt_classes, list) or isinstance(gt_classes, tuple):
                    gt_classes = [gt_classes]
                gt_classes = augment_gt_classes_with_plural_and_singular(gt_classes)
                # A few masks are too small to be kept in 1120 x 1120 resolution, we remove these cases from evaluation.
                if "open_end_pred" not in segment_info:
                    continue
                pred_class = segment_info["open_end_pred"]
                pred_class_cityscapes = pred_class["cityscapes"]["class"].lower()
                pred_class_any = pred_class["any"]["class"].lower()

                total_sample += 1
                corr_sample_any += int(pred_class_any in gt_classes)
                corr_sample_cityscapes += int(pred_class_cityscapes in gt_classes)
                not_in_vocab_any += int(pred_class_any not in all_cityscapes_vocabularies)
                not_in_vocab_cityscapes += int(pred_class_cityscapes not in all_cityscapes_vocabularies)


    print(f"\nCityscapes Panoptic evaluation:")
    print(f"Agnostic: Acc: {corr_sample_any}/{total_sample} = {corr_sample_any/total_sample}, Not-In-Vocab: {not_in_vocab_any}/{total_sample} = {not_in_vocab_any/total_sample}")
    print(f"Specific: Acc: {corr_sample_cityscapes}/{total_sample} = {corr_sample_cityscapes/total_sample}, Not-In-Vocab: {not_in_vocab_cityscapes}/{total_sample} = {not_in_vocab_cityscapes/total_sample}")



# LVIS Instance
def get_all_lvis_class():
    lvis_meta = openseg_classes.get_lvis_categories_with_prompt_eng()
    lvis_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": x["isthing"]} for x in lvis_meta}
    return lvis_class
all_lvis_class = get_all_lvis_class()
def lvis_id2class(idx):
    return all_lvis_class[idx]["name"]

all_lvis_vocabularies = []
for x in all_lvis_class:
    all_lvis_vocabularies += all_lvis_class[x]["name"]
# convert to dict to speed-up
all_lvis_vocabularies = {x:1 for x in all_lvis_vocabularies}

def parse_lvis_instance(json_file_list):
    total_sample = 0
    corr_sample_any = 0
    corr_sample_lvis = 0
    not_in_vocab_any = 0
    not_in_vocab_lvis = 0

    for json_file in json_file_list:
        json_log = json.load(open(json_file))["pred"]
        for img_log in json_log:
            img_dict, anno_dict_list = img_log
            for anno_dict in anno_dict_list:
                if anno_dict.get("iscrowd", 0):
                    continue
                gt_classes = lvis_id2class(anno_dict["category_id"])
                if not isinstance(gt_classes, list) or isinstance(gt_classes, tuple):
                    gt_classes = [gt_classes]
                gt_classes = augment_gt_classes_with_plural_and_singular(gt_classes)
                # A few masks are too small to be kept in 1120 x 1120 resolution, we remove these cases from evaluation.
                if "open_end_pred" not in anno_dict:
                    # example: {'area': 0.44, 'id': 57394, 'segmentation': [[257.24, 459.31, 256.5, 459.97, 257.02, 460.1, 257.28, 460.1, 257.55, 459.71, 257.24, 459.31]], 'image_id': 163589, 'bbox': [256.5, 459.31, 1.05, 0.79], 'category_id': 338}
                    continue
                pred_class = anno_dict["open_end_pred"]
                pred_class_lvis = pred_class["lvis"]["class"].lower()
                pred_class_any = pred_class["any"]["class"].lower()

                total_sample += 1
                corr_sample_any += int(pred_class_any in gt_classes)
                corr_sample_lvis += int(pred_class_lvis in gt_classes)
                not_in_vocab_any += int(pred_class_any not in all_lvis_vocabularies)
                not_in_vocab_lvis += int(pred_class_lvis not in all_lvis_vocabularies)

    print(f"\nLVIS Instance evaluation:")
    print(f"Agnostic: Acc: {corr_sample_any}/{total_sample} = {corr_sample_any/total_sample}, Not-In-Vocab: {not_in_vocab_any}/{total_sample} = {not_in_vocab_any/total_sample}")
    print(f"Specific: Acc: {corr_sample_lvis}/{total_sample} = {corr_sample_lvis/total_sample}, Not-In-Vocab: {not_in_vocab_lvis}/{total_sample} = {not_in_vocab_lvis/total_sample}")


# A-847 Semantic
def get_all_a847_class():
    a847_meta = openseg_classes.get_ade20k_847_categories_with_prompt_eng()
    a847_class = {x["trainId"]: {"name": x["name"].lower().split(","), "isthing": 0} for x in a847_meta}
    #print("a847_class:", a847_class)
    return a847_class
all_a847_class = get_all_a847_class()
def a847_id2class(idx):
    return all_a847_class[idx]["name"]
all_a847_vocabularies = []
for x in all_a847_class:
    all_a847_vocabularies += all_a847_class[x]["name"]
# convert to dict to speed-up
all_a847_vocabularies = {x:1 for x in all_a847_vocabularies}

def parse_a847_semantic(json_file_list):
    total_sample = 0
    corr_sample_any = 0
    corr_sample_a847 = 0

    corr_sample_any = 0
    corr_sample_a847 = 0
    not_in_vocab_any = 0
    not_in_vocab_a847 = 0

    for json_file in json_file_list:
        json_log = json.load(open(json_file))["pred"]
        for img_log in json_log:
            pred = img_log["pred"]
            for pred_info in pred:
                gt_classes = a847_id2class(pred_info["mask_unique_id"])
                if not isinstance(gt_classes, list) or isinstance(gt_classes, tuple):
                    gt_classes = [gt_classes]
                gt_classes = augment_gt_classes_with_plural_and_singular(gt_classes)

                pred_class = pred_info["open_end_pred"]
                pred_class_a847 = pred_class["a847"]["class"].lower()
                pred_class_any = pred_class["any"]["class"].lower()

                total_sample += 1
                corr_sample_any += int(pred_class_any in gt_classes)
                corr_sample_a847 += int(pred_class_a847 in gt_classes)
                not_in_vocab_any += int(pred_class_any not in all_a847_vocabularies)
                not_in_vocab_a847 += int(pred_class_a847 not in all_a847_vocabularies)

    print(f"\nA-847 Semantic evaluation:")
    print(f"Agnostic: Acc: {corr_sample_any}/{total_sample} = {corr_sample_any/total_sample}, Not-In-Vocab: {not_in_vocab_any}/{total_sample} = {not_in_vocab_any/total_sample}")
    print(f"Specific: Acc: {corr_sample_a847}/{total_sample} = {corr_sample_a847/total_sample}, Not-In-Vocab: {not_in_vocab_a847}/{total_sample} = {not_in_vocab_a847/total_sample}")


# PC-459 Semantic
def get_all_pc459_class():
    pc459_meta = openseg_classes.get_pascal_ctx_459_categories_with_prompt_eng()
    pc459_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": 0} for x in pc459_meta}
    #print("pc459_class:", pc459_class)
    return pc459_class
all_pc459_class = get_all_pc459_class()
def pc459_id2class(idx):
    return all_pc459_class[idx]["name"]

all_pc459_vocabularies = []
for x in all_pc459_class:
    all_pc459_vocabularies += all_pc459_class[x]["name"]
# convert to dict to speed-up
all_pc459_vocabularies = {x:1 for x in all_pc459_vocabularies}

def parse_pc459_semantic(json_file_list):
    total_sample = 0
    corr_sample_any = 0
    corr_sample_pc459 = 0

    corr_sample_any = 0
    corr_sample_pc459 = 0
    not_in_vocab_any = 0
    not_in_vocab_pc459 = 0

    for json_file in json_file_list:
        json_log = json.load(open(json_file))["pred"]
        for img_log in json_log:
            pred = img_log["pred"]
            for pred_info in pred:
                gt_classes = pc459_id2class(pred_info["mask_unique_id"])
                if not isinstance(gt_classes, list) or isinstance(gt_classes, tuple):
                    gt_classes = [gt_classes]
                gt_classes = augment_gt_classes_with_plural_and_singular(gt_classes)

                pred_class = pred_info["open_end_pred"]
                pred_class_pc459 = pred_class["pc459"]["class"].lower()
                pred_class_any = pred_class["any"]["class"].lower()

                total_sample += 1
                corr_sample_any += int(pred_class_any in gt_classes)
                corr_sample_pc459 += int(pred_class_pc459 in gt_classes)
                not_in_vocab_any += int(pred_class_any not in all_pc459_vocabularies)
                not_in_vocab_pc459 += int(pred_class_pc459 not in all_pc459_vocabularies)

    print(f"\nPC-459 Semantic evaluation:")
    print(f"Agnostic: Acc: {corr_sample_any}/{total_sample} = {corr_sample_any/total_sample}, Not-In-Vocab: {not_in_vocab_any}/{total_sample} = {not_in_vocab_any/total_sample}")
    print(f"Specific: Acc: {corr_sample_pc459}/{total_sample} = {corr_sample_pc459/total_sample}, Not-In-Vocab: {not_in_vocab_pc459}/{total_sample} = {not_in_vocab_pc459/total_sample}")


if __name__ == "__main__":
    import glob
    import sys
    save_path = str(sys.argv[1])
    total_split = int(sys.argv[2])
    coco_json_list = glob.glob(f"{save_path}/coco_pred_*_of_{total_split}.json")
    assert len(coco_json_list) == total_split
    parse_coco_panoptic(coco_json_list)

    ade20k_json_list = glob.glob(f"{save_path}/ade20k_pred_*_of_{total_split}.json")
    assert len(ade20k_json_list) == total_split
    parse_ade20k_panoptic(ade20k_json_list)

    cityscapes_json_list = glob.glob(f"{save_path}/cityscapes_pred_*_of_{total_split}.json")
    assert len(cityscapes_json_list) == total_split
    parse_cityscapes_panoptic(cityscapes_json_list)

    lvis_json_list = glob.glob(f"{save_path}/lvis_pred_*_of_{total_split}.json")
    assert len(lvis_json_list) == total_split
    parse_lvis_instance(lvis_json_list)

    a847_json_list = glob.glob(f"{save_path}/a847_pred_*_of_{total_split}.json")
    assert len(a847_json_list) == total_split
    parse_a847_semantic(a847_json_list)

    pc459_json_list = glob.glob(f"{save_path}/pc459_pred_*_of_{total_split}.json")
    assert len(pc459_json_list) == total_split
    parse_pc459_semantic(pc459_json_list)


"""
Testing with osm_final.pt in the model zoo gives the following results (slightly different from the ones in paper due to float16 testing):

COCO Panoptic evaluation:
Agnostic: Acc: 44738/56295 = 0.7947064570565769, Not-In-Vocab: 4940/56295 = 0.08775202060573763
Specific: Acc: 48953/56295 = 0.8695798916422418, Not-In-Vocab: 62/56295 = 0.00110134114930278

ADE20K Panoptic evaluation:
Agnostic: Acc: 25333/30250 = 0.8374545454545455, Not-In-Vocab: 641/30250 = 0.0211900826446281
Specific: Acc: 25770/30250 = 0.851900826446281, Not-In-Vocab: 94/30250 = 0.0031074380165289255

Cityscapes Panoptic evaluation:
Agnostic: Acc: 12748/14373 = 0.8869407917623322, Not-In-Vocab: 2/14373 = 0.00013914979475405275
Specific: Acc: 12741/14373 = 0.8864537674806929, Not-In-Vocab: 2/14373 = 0.00013914979475405275

LVIS Instance evaluation:
Agnostic: Acc: 157898/244645 = 0.6454168284657361, Not-In-Vocab: 20250/244645 = 0.08277299760878007
Specific: Acc: 177243/244645 = 0.7244905884036053, Not-In-Vocab: 2878/244645 = 0.011763984549040447

A-847 Semantic evaluation:
Agnostic: Acc: 15591/20369 = 0.765427856055771, Not-In-Vocab: 164/20369 = 0.008051450733958467
Specific: Acc: 15911/20369 = 0.7811380038293485, Not-In-Vocab: 100/20369 = 0.0049094211792429674

PC-459 Semantic evaluation:
Agnostic: Acc: 25087/31137 = 0.8056974018049267, Not-In-Vocab: 1163/31137 = 0.03735106143816039
Specific: Acc: 25847/31137 = 0.8301056620740598, Not-In-Vocab: 167/31137 = 0.005363394032822687
"""
