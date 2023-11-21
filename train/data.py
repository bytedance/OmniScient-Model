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

Reference: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/data.py
"""

import functools
import io
import json
import math
import random
import numpy as np
import torch
import webdataset as wds
from PIL import Image
import base64
import torchvision.transforms as T

from data_utils import *
from dataset_names import openseg_classes
import pycocotools.mask as mask_util
import copy

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 10
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

# Hyper-params introduced by OSM
from utils import get_context_mask
from panopticapi.utils import rgb2id
LOWER_CHAR = True
MAX_RETRY_CNT = 500
MIN_MASK_SIZE = 100
MASK_PROMPT = [
    "What is in the segmentation mask? Assistant:",
    "Describe what is in the segmentation mask. Assistant:",
    "What does this segmentation mask show? Assistant:",
    "What is this segmentation mask? Assistant:",
    "What is the segmentation mask region of the image? Assistant:",
    "Briefly describe what you perceive in the segmentation mask region. Assistant:",
    "Please tell the category of what is indicated by the segmentation mask. Assistant:",
    "What does this segmentation mask segments? Assistant:",
    "What does this segmentation mask capture? Assistant:",
    "Answer the name of what is in the segmentation mask region. Assistant:",
    "What is the semantic class of the area given the segmentation mask? Assistant:",
    "Can you describe what is in the segmentation mask region? Assistant:",
    "From the image and segmentation mask provided, tell the category of the indicated region. Assistant:",
    "Could you use a few words to describe what is in the segmentation mask region? Assistant:",
    "Given the image and segmentation mask, answer what is in the region. Assistant:",
    "Tell me what you see in the segmentation mask region. Assistant:",
    "What can you see in the segmentation mask region? Assistant:",
    "Let me know what you can perceive in the mask region. Assistant:",
    "Give me the name of the object in the segmentation mask. Assistant:"]
BBOX_PROMPT = [
    "What is in the bounding box? Assistant:",
    "Describe what is in the bounding box. Assistant:",
    "What does this bounding box show? Assistant:",
    "What is this bounding box? Assistant:",
    "What is the bounding box region of the image? Assistant:",
    "Briefly describe what you perceive in the bounding box region. Assistant:",
    "Please tell the category of what is indicated by the bounding box. Assistant:",
    "What does this bounding box segments? Assistant:",
    "What does this bounding box capture? Assistant:",
    "Answer the name of what is in the bounding box region. Assistant:",
    "What is the semantic class of the area given the bounding box? Assistant:",
    "Can you describe what is in the bounding box region? Assistant:",
    "From the image and bounding box provided, tell the category of the indicated region. Assistant:",
    "Could you use a few words to describe what is in the bounding box region? Assistant:",
    "Given the image and bounding box, answer what is in the region. Assistant:",
    "Tell me what you see in the bounding box region. Assistant:",
    "What can you see in the bounding box region? Assistant:",
    "Let me know what you can perceive in the box region. Assistant:",
    "Give me the name of the object in the bounding box. Assistant:"]


def get_dataset(args, processor, dataset_type, epoch=0, floor=False):
    input_shards = {
        "coco": args.coco_shards,
        "lvis": args.lvis_shards,
        "v3det": args.v3det_shards,
        "a847": args.a847_shards,
        "pc459": args.pc459_shards,
        "partimagenet": args.partimagenet_shards,
        "pascal_part": args.pascal_part_shards,
        "ade20k": args.ade20k_shards,
        "cityscapes": args.cityscapes_shards
        }[dataset_type]
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = {
            "coco": args.train_num_samples_coco,
            "lvis": args.train_num_samples_lvis,
            "v3det": args.train_num_samples_v3det,
            "a847": args.train_num_samples_a847,
            "pc459": args.train_num_samples_pc459,
            "partimagenet": args.train_num_samples_partimagenet,
            "pascal_part": args.train_num_samples_pascal_part,
            "ade20k": args.train_num_samples_ade20k,
            "cityscapes": args.train_num_samples_cityscapes
            }[dataset_type]
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )
    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_fn = {
        "coco": preprocess_coco,
        "lvis": preprocess_lvis,
        "v3det": preprocess_v3det,
        "a847": preprocess_a847,
        "pc459": preprocess_pc459,
        "partimagenet": preprocess_partimagenet,
        "pascal_part": preprocess_pascal_part,
        "ade20k": preprocess_ade20k,
        "cityscapes": preprocess_cityscapes
        }[dataset_type]

    if dataset_type in ["coco", "ade20k", "cityscapes", "lvis"]:
        preprocess_fn = functools.partial(
            preprocess_fn,
            processor=processor,
            max_tokens=32,
            input_size=args.input_size,
            random_scale_min=args.random_scale_min,
            random_scale_max=args.random_scale_max,
            mask2box_prob=args.mask2box_prob,
            context_enlarge_ratio=args.context_enlarge_ratio,
        )
    elif dataset_type in ["v3det", "a847", "pc459", "pascal_part"]:
        preprocess_fn = functools.partial(
            preprocess_fn,
            processor=processor,
            max_tokens=32,
            input_size=args.input_size,
            random_scale_min=args.random_scale_min,
            random_scale_max=args.random_scale_max,
            context_enlarge_ratio=args.context_enlarge_ratio,
        )
    elif dataset_type in ["partimagenet"]:
        preprocess_fn = functools.partial(
            preprocess_fn,
            processor=processor,
            max_tokens=32,
            input_size=args.input_size,
            random_scale_min=args.random_scale_min,
            random_scale_max=args.random_scale_max,
            context_enlarge_ratio=args.context_enlarge_ratio,
            part_in_use_whole_prob=args.part_in_use_whole_prob,
        part_in_prepend_object_class_prob=args.part_in_prepend_object_class_prob
        )

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_coco, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    batch_size = {
        "coco": args.batch_size_coco,
        "lvis": args.batch_size_lvis,
        "v3det": args.batch_size_v3det,
        "a847": args.batch_size_a847,
        "pc459": args.batch_size_pc459,
        "partimagenet": args.batch_size_partimagenet,
        "pascal_part": args.batch_size_pascal_part,
        "ade20k": args.batch_size_ade20k,
        "cityscapes": args.batch_size_cityscapes
        }[dataset_type]
    global_batch_size = batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(dataset_type):
    return functools.partial(get_dataset, dataset_type=dataset_type)

def get_data(args, processor, dataset_type, epoch=0):
    """
    Interface for getting the webdatasets
    """
    return get_dataset_fn(dataset_type)(
        args, processor=processor, epoch=epoch
    )

# This is shared for all panoptic datasets in COCO format
def parse_sample_coco(sample, processor, max_tokens, id2isthing_func, id2class_func, dataset_name,
                      input_size, random_scale_min, random_scale_max, mask2box_prob, context_enlarge_ratio):
    info = json.loads(sample[0])
    segments_info = info["segments_info"]
    image = info["image_base64"]
    rawbytes = base64.b64decode(image)
    image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
    pan_seg = info["pan_base64"]
    rawbytes = base64.b64decode(pan_seg)
    pan_seg = Image.open(io.BytesIO(rawbytes)).convert("RGB")

    mask_prompt = random.choice(MASK_PROMPT)
    bbox_prompt = random.choice(BBOX_PROMPT)
    scale_factor = np.random.uniform(random_scale_min, random_scale_max)

    if min(image.size) < input_size:
        target_size = int(scale_factor * input_size)
        if min(image.size) == max(image.size):
            image = T.functional.resize(image, size=target_size,
                                    interpolation=T.functional.InterpolationMode.BICUBIC)
            pan_seg = T.functional.resize(pan_seg, size=target_size,
                                    interpolation=T.functional.InterpolationMode.NEAREST)
        else:
            image = T.functional.resize(image, size=target_size-1, max_size=target_size,
                                        interpolation=T.functional.InterpolationMode.BICUBIC)
            pan_seg = T.functional.resize(pan_seg, size=target_size-1, max_size=target_size,
                                    interpolation=T.functional.InterpolationMode.NEAREST)
    image = np.array(image)
    pan_seg = rgb2id(np.array(pan_seg)) # H x W
    # random flip
    if torch.rand(1) < 0.5:
        image = image[:, ::-1]
        pan_seg = pan_seg[:, ::-1]
    try_cnt = 0
    while True:
        random_h = np.random.randint(0, max(0, pan_seg.shape[0] - input_size) + 1)
        random_w = np.random.randint(0, max(0, pan_seg.shape[1] - input_size) + 1)
        image_ = image[random_h:random_h+input_size, random_w:random_w+input_size, :].copy()
        pan_seg_ = pan_seg[random_h:random_h+input_size, random_w:random_w+input_size].copy()
        # padding to input_size
        padded_image = np.zeros(shape=(input_size, input_size, 3), dtype=np.uint8)
        padded_pan_seg = -np.ones(shape=(input_size, input_size), dtype=np.int32)
        padded_image[:image_.shape[0], :image_.shape[1]] = image_
        padded_pan_seg[:pan_seg_.shape[0], :pan_seg_.shape[1]] = pan_seg_
        image_ = padded_image
        pan_seg_ = padded_pan_seg

        # next, we generate masks
        masks = []
        classes = []
        prompts = []

        for segment_info in segments_info:
            if segment_info.get("iscrowd", 0):
                continue
            binary_mask = (pan_seg_ == segment_info["id"])
            if binary_mask.sum() <= MIN_MASK_SIZE:
                continue
            
            # only apply to thing
            if id2isthing_func(segment_info["category_id"]) and random.random() < mask2box_prob:
                # convert the mask to a bounding box style mask
                all_idx = np.nonzero(binary_mask)
                minh = min(all_idx[0])
                maxh = max(all_idx[0]) + 1
                minw = min(all_idx[1])
                maxw = max(all_idx[1]) + 1
                bbox_size_h = maxh - minh
                bbox_size_w = maxw - minw
                if bbox_size_h * bbox_size_w <= MIN_MASK_SIZE:
                    continue
                binary_mask[minh:maxh, minw:maxw] = 1
                prompts.append(bbox_prompt)
            else:
                prompts.append(mask_prompt)
            
            masks.append(binary_mask)
            class_name = id2class_func(segment_info["category_id"])[0]
            classes.append(class_name)

        if len(masks) > 0:
            pan_seg = pan_seg_.copy()
            image = image_.copy()
            break
        try_cnt += 1
        if try_cnt >= MAX_RETRY_CNT:
            # this will be catched and retry with another sample
            raise ValueError(f"{dataset_name}: try_cnt larger than {MAX_RETRY_CNT}")
        
    image = Image.fromarray(image)
    processor.image_processor.size = {"height": input_size, "width": input_size}
    image = processor(images=image, return_tensors="pt")["pixel_values"]
    # shuffle masks and classes
    all_idx = list(range(len(masks)))
    random.shuffle(all_idx)
    # We just keep the first mask + class
    input_mask = masks[all_idx[0]].reshape(1, input_size, input_size)
    target_class = classes[all_idx[0]].strip()
    prompt = prompts[all_idx[0]]
    final_text = prompt + ' ' + target_class
    processor.tokenizer.padding_side = "right"
    processor.qformer_tokenizer.padding_side = "right"
    text_tensor = processor(
        text=final_text + processor.tokenizer.eos_token,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    instruction = processor(
        text=prompt,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_mask = torch.from_numpy(np.ascontiguousarray(input_mask.copy()))
    input_context_mask = get_context_mask(input_mask, input_size=input_size, enlarge_ratio=context_enlarge_ratio)

    return (
        image, input_mask, input_context_mask, text_tensor["input_ids"], text_tensor["attention_mask"],
         instruction["qformer_input_ids"], instruction["qformer_attention_mask"])

def get_all_coco_class():
        coco_meta = openseg_classes.get_coco_categories_with_prompt_eng()
        if LOWER_CHAR:
            coco_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": x["isthing"]} for x in coco_meta}
        else:
            coco_class = {x["id"]: {"name": x["name"].split(","), "isthing": x["isthing"]} for x in coco_meta}
        return coco_class

all_coco_class = get_all_coco_class()

def preprocess_coco(
    sample,
    processor,
    max_tokens,
    input_size,
    random_scale_min,
    random_scale_max,
    mask2box_prob,
    context_enlarge_ratio
):
    
    def coco_id2class(idx, all_coco_class):
        return all_coco_class[idx]["name"]

    def coco_id2isthing(idx, all_coco_class):
        return all_coco_class[idx]["isthing"]
    
    return parse_sample_coco(sample, processor, max_tokens,
                             id2isthing_func=functools.partial(coco_id2isthing, all_coco_class=all_coco_class),
                             id2class_func=functools.partial(coco_id2class, all_coco_class=all_coco_class),
                             dataset_name="COCO Panoptic", input_size=input_size,
                             random_scale_min=random_scale_min, random_scale_max=random_scale_max,
                             mask2box_prob=mask2box_prob, context_enlarge_ratio=context_enlarge_ratio)


def get_all_ade20k_class():
        ade20k_meta = openseg_classes.get_ade20k_categories_with_prompt_eng()
        if LOWER_CHAR:
            ade20k_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": x["isthing"]} for x in ade20k_meta}
        else:
            ade20k_class = {x["id"]: {"name": x["name"].split(","), "isthing": x["isthing"]} for x in ade20k_meta}
        return ade20k_class

all_ade20k_class = get_all_ade20k_class()

def preprocess_ade20k(
    sample,
    processor,
    max_tokens,
    input_size,
    random_scale_min,
    random_scale_max,
    mask2box_prob,
    context_enlarge_ratio
):
    
    def ade20k_id2class(idx, all_ade20k_class):
        return all_ade20k_class[idx]["name"]

    def ade20k_id2isthing(idx, all_ade20k_class):
        return all_ade20k_class[idx]["isthing"]

    
    return parse_sample_coco(sample, processor, max_tokens,
                              id2isthing_func=functools.partial(ade20k_id2isthing, all_ade20k_class=all_ade20k_class),
                              id2class_func=functools.partial(ade20k_id2class, all_ade20k_class=all_ade20k_class),
                              dataset_name="ADE20K Panoptic", input_size=input_size,
                              random_scale_min=random_scale_min, random_scale_max=random_scale_max,
                              mask2box_prob=mask2box_prob, context_enlarge_ratio=context_enlarge_ratio)


def get_all_cityscapes_class():
        cityscapes_meta = openseg_classes.get_cityscapes_categories_with_prompt_eng()
        if LOWER_CHAR:
            cityscapes_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": x["isthing"]} for x in cityscapes_meta}
        else:
            cityscapes_class = {x["id"]: {"name": x["name"].split(","), "isthing": x["isthing"]} for x in cityscapes_meta}
        return cityscapes_class

all_cityscapes_class = get_all_cityscapes_class()

def preprocess_cityscapes(
    sample,
    processor,
    max_tokens,
    input_size,
    random_scale_min,
    random_scale_max,
    mask2box_prob,
    context_enlarge_ratio
):
    
    def cityscapes_id2class(idx, all_cityscapes_class):
        return all_cityscapes_class[idx]["name"]

    def cityscapes_id2isthing(idx, all_cityscapes_class):
        return all_cityscapes_class[idx]["isthing"]

    
    return parse_sample_coco(sample, processor, max_tokens,
                              id2isthing_func=functools.partial(cityscapes_id2isthing, all_cityscapes_class=all_cityscapes_class),
                              id2class_func=functools.partial(cityscapes_id2class, all_cityscapes_class=all_cityscapes_class),
                              dataset_name="Cityscapes Panoptic", input_size=input_size,
                              random_scale_min=random_scale_min, random_scale_max=random_scale_max,
                              mask2box_prob=mask2box_prob, context_enlarge_ratio=context_enlarge_ratio)

# This is shared for all instance datasets in LVIS format
def parse_sample_lvis(sample, processor, max_tokens, id2isthing_func, id2class_func, dataset_name, 
                      input_size, random_scale_min, random_scale_max, mask2box_prob, context_enlarge_ratio, decode_mask=True):
    info = json.loads(sample[0])
    # a list of dict, which contains bbox, segmentation (polygon), and category_id
    segments_info = info["segments_info"]
    image = info["image_base64"]
    height, width = info["height"], info["width"]
    rawbytes = base64.b64decode(image)
    image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
    mask_prompt = random.choice(MASK_PROMPT)
    bbox_prompt = random.choice(BBOX_PROMPT)

    scale_factor = np.random.uniform(random_scale_min, random_scale_max)

    if min(image.size) < input_size:
        target_size = int(scale_factor * input_size)
        if min(image.size) == max(image.size):
            image = T.functional.resize(image, size=target_size,
                                    interpolation=T.functional.InterpolationMode.BICUBIC)
        else:
            image = T.functional.resize(image, size=target_size-1, max_size=target_size,
                                        interpolation=T.functional.InterpolationMode.BICUBIC)
    image = np.array(image)

    # random flip
    do_flip = torch.rand(1) < 0.5
    if do_flip:
        image = image[:, ::-1]

    try_cnt = -1
    while True:
        try_cnt += 1
        if try_cnt >= MAX_RETRY_CNT:
            raise ValueError(f"{dataset_name}: try_cnt larger than {MAX_RETRY_CNT}")
        # We randomly pick a mask at the beginning, to avoid costs decoding polygon to mask
        random_idx = random.randint(0, len(segments_info) - 1)
        category_id = segments_info[random_idx]["category_id"]
        if segments_info[random_idx].get("iscrowd", 0):
            continue
        if decode_mask:
            # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/structures/masks.html
            segmentation_polygon = segments_info[random_idx]["segmentation"]
            rles = mask_util.frPyObjects(segmentation_polygon, height, width)
            rle = mask_util.merge(rles)
            binary_mask = mask_util.decode(rle).astype(bool)
            binary_mask = Image.fromarray(binary_mask.astype(np.uint8))
        else:
            bbox = copy.deepcopy(segments_info[random_idx]["bbox"])
            # BoxMode.XYWH_ABS -> XYXY_ABS
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            # round to int
            x1, y1, x2, y2 = bbox
            x1, y1 = int(math.floor(x1)), int(math.floor(y1))
            x2, y2 = int(math.ceil(x2)), int(math.ceil(y2))
            y1 = max(0, y1)
            x1 = max(0, x1)
            x2 = min(x2, width)
            y2 = min(y2, height)
            # convert to mask
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            binary_mask[y1:y2, x1:x2] = 1
            binary_mask = Image.fromarray(binary_mask)

        if min(height, width) < input_size:
            if height == width:
                binary_mask = T.functional.resize(binary_mask, size=target_size,
                                                interpolation=T.functional.InterpolationMode.NEAREST)
            else:
                binary_mask = T.functional.resize(binary_mask, size=target_size-1, max_size=target_size,
                                                interpolation=T.functional.InterpolationMode.NEAREST)
        binary_mask = np.array(binary_mask)
        assert len(binary_mask.shape) == 2

        if do_flip:
            binary_mask = binary_mask[:, ::-1]

        # random crop
        random_h = np.random.randint(0, max(0, binary_mask.shape[0] - input_size) + 1)
        random_w = np.random.randint(0, max(0, binary_mask.shape[1] - input_size) + 1)
        image_ = image[random_h:random_h+input_size, random_w:random_w+input_size, :].copy()
        binary_mask_ = binary_mask[random_h:random_h+input_size, random_w:random_w+input_size].copy()

        # pad to INPUT_SIZE
        padded_image = np.zeros(shape=(input_size, input_size, 3), dtype=np.uint8)
        padded_image[:image_.shape[0], :image_.shape[1]] = image_
        image_ = padded_image
        padded_binary_mask = np.zeros(shape=(input_size, input_size), dtype=np.uint8)
        padded_binary_mask[:binary_mask_.shape[0], :binary_mask_.shape[1]] = binary_mask_
        binary_mask_ = padded_binary_mask

        if binary_mask_.sum() <= MIN_MASK_SIZE:
            continue

        # only apply to thing
        if (not decode_mask) or (
            id2isthing_func(category_id) and random.random() < mask2box_prob):
            # convert the mask to a bounding box style mask
            all_idx = np.nonzero(binary_mask_)
            minh = min(all_idx[0])
            maxh = max(all_idx[0]) + 1
            minw = min(all_idx[1])
            maxw = max(all_idx[1]) + 1
            bbox_size_h = maxh - minh
            bbox_size_w = maxw - minw
            if bbox_size_h * bbox_size_w <= MIN_MASK_SIZE:
                continue
            binary_mask_[minh:maxh, minw:maxw] = 1
            prompt = bbox_prompt
        else:
            prompt = mask_prompt

        target_class = id2class_func(category_id)[0].strip()
        image = image_.copy()
        break
        
    image = Image.fromarray(image)
    processor.image_processor.size = {"height": input_size, "width": input_size}
    image = processor(images=image, return_tensors="pt")["pixel_values"]

    input_mask = binary_mask_.reshape(1, input_size, input_size)
    final_text = prompt + ' ' + target_class

    processor.tokenizer.padding_side = "right"
    processor.qformer_tokenizer.padding_side = "right"
    text_tensor = processor(
        text=final_text + processor.tokenizer.eos_token,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    instruction = processor(
        text=prompt,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_mask = torch.from_numpy(np.ascontiguousarray(input_mask.copy()))
    input_context_mask = get_context_mask(input_mask, input_size=input_size, enlarge_ratio=context_enlarge_ratio)

    return (
        image, input_mask, input_context_mask, text_tensor["input_ids"], text_tensor["attention_mask"],
         instruction["qformer_input_ids"], instruction["qformer_attention_mask"])


def get_all_lvis_class():
        lvis_meta = openseg_classes.get_lvis_categories_with_prompt_eng()
        if LOWER_CHAR:
            lvis_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": x["isthing"]} for x in lvis_meta}
        else:
            lvis_class = {x["id"]: {"name": x["name"].split(","), "isthing": x["isthing"]} for x in lvis_meta}
        return lvis_class

all_lvis_class = get_all_lvis_class()

def preprocess_lvis(
    sample,
    processor,
    max_tokens,
    input_size,
    random_scale_min,
    random_scale_max,
    mask2box_prob,
    context_enlarge_ratio
):
    
    def lvis_id2class(idx, all_lvis_class):
        return all_lvis_class[idx]["name"]

    def lvis_id2isthing(idx, all_lvis_class):
        return all_lvis_class[idx]["isthing"]

    return parse_sample_lvis(sample, processor, max_tokens,
                              id2isthing_func=functools.partial(lvis_id2isthing, all_lvis_class=all_lvis_class),
                              id2class_func=functools.partial(lvis_id2class, all_lvis_class=all_lvis_class),
                              dataset_name="LVIS Instance", input_size=input_size,
                              random_scale_min=random_scale_min, random_scale_max=random_scale_max,
                              mask2box_prob=mask2box_prob, context_enlarge_ratio=context_enlarge_ratio,
                              decode_mask=True)

def get_all_v3det_class():
        v3det_meta = openseg_classes.get_v3det_categories()
        if LOWER_CHAR:
            v3det_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": x["isthing"]} for x in v3det_meta}
        else:
            v3det_class = {x["id"]: {"name": x["name"].split(","), "isthing": x["isthing"]} for x in v3det_meta}
        return v3det_class

all_v3det_class = get_all_v3det_class()

def preprocess_v3det(
    sample,
    processor,
    max_tokens,
    input_size,
    random_scale_min,
    random_scale_max,
    context_enlarge_ratio
):
    
    def v3det_id2class(idx, all_v3det_class):
        return all_v3det_class[idx]["name"]
    
    def v3det_id2isthing(idx, all_v3det_class):
        return all_v3det_class[idx]["isthing"]

    return parse_sample_lvis(sample, processor, max_tokens,
                              id2isthing_func=functools.partial(v3det_id2isthing, all_v3det_class=all_v3det_class),
                              id2class_func=functools.partial(v3det_id2class, all_v3det_class=all_v3det_class),
                              dataset_name="V3Det Detection", input_size=input_size,
                              random_scale_min=random_scale_min, random_scale_max=random_scale_max,
                              mask2box_prob=0, context_enlarge_ratio=context_enlarge_ratio,
                              decode_mask=False)

# This is shared for all semantic datasets in A-847 format
def parse_sample_a847(sample, processor, max_tokens, id2class_func, dataset_name,
                      ignore_value_list, input_size, random_scale_min, random_scale_max, context_enlarge_ratio):
    info = json.loads(sample[0])
    image = info["image_base64"]
    rawbytes = base64.b64decode(image)
    image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
    sem_seg = info["seg_base64"]
    rawbytes = base64.b64decode(sem_seg)
    sem_seg = Image.open(io.BytesIO(rawbytes))

    prompt = random.choice(MASK_PROMPT)
    scale_factor = np.random.uniform(random_scale_min, random_scale_max)

    if min(image.size) < input_size:
        target_size = int(scale_factor * input_size)
        if min(image.size) == max(image.size):
            image = T.functional.resize(image, size=target_size,
                                    interpolation=T.functional.InterpolationMode.BICUBIC)
            sem_seg = T.functional.resize(sem_seg, size=target_size,
                                    interpolation=T.functional.InterpolationMode.NEAREST)
        else:
            image = T.functional.resize(image, size=target_size-1, max_size=target_size,
                                        interpolation=T.functional.InterpolationMode.BICUBIC)
            sem_seg = T.functional.resize(sem_seg, size=target_size-1, max_size=target_size,
                                    interpolation=T.functional.InterpolationMode.NEAREST)
    image = np.array(image)
    sem_seg = np.array(sem_seg) # H x W
    # random flip
    if torch.rand(1) < 0.5:
        image = image[:, ::-1]
        sem_seg = sem_seg[:, ::-1]

    try_cnt = 0
    while True:
        random_h = np.random.randint(0, max(0, sem_seg.shape[0] - input_size) + 1)
        random_w = np.random.randint(0, max(0, sem_seg.shape[1] - input_size) + 1)
        image_ = image[random_h:random_h+input_size, random_w:random_w+input_size, :].copy()
        sem_seg_ = sem_seg[random_h:random_h+input_size, random_w:random_w+input_size].copy()
        # padding to input_size
        padded_image = np.zeros(shape=(input_size, input_size, 3), dtype=np.uint8)
        padded_sem_seg = np.ones(shape=(input_size, input_size), dtype=np.int32) * ignore_value_list[0]
        padded_image[:image_.shape[0], :image_.shape[1]] = image_
        padded_sem_seg[:sem_seg_.shape[0], :sem_seg_.shape[1]] = sem_seg_
        image_ = padded_image
        sem_seg_ = padded_sem_seg

        # next, we generate masks
        masks = []
        classes = []

        all_ids = np.unique(sem_seg_)
        for unique_id in all_ids:
            if unique_id in ignore_value_list:
                continue
            binary_mask = (sem_seg_ == unique_id)
            if binary_mask.sum() <= MIN_MASK_SIZE:
                continue

            masks.append(binary_mask)
            class_name = id2class_func(unique_id)[0]
            classes.append(class_name)

        if len(masks) > 0:
            sem_seg = sem_seg_.copy()
            image = image_.copy()
            break
        try_cnt += 1
        if try_cnt >= MAX_RETRY_CNT:
            # this will be catched and retry with another sample
            raise ValueError(f"{dataset_name}: try_cnt larger than {MAX_RETRY_CNT}")
        
    image = Image.fromarray(image)
    processor.image_processor.size = {"height": input_size, "width": input_size}
    image = processor(images=image, return_tensors="pt")["pixel_values"]
    # shuffle masks and classes
    all_idx = list(range(len(masks)))
    random.shuffle(all_idx)
    # We just keep the first mask + class
    input_mask = masks[all_idx[0]].reshape(1, input_size, input_size)
    target_class = classes[all_idx[0]].strip()
    final_text = prompt + ' ' + target_class
    processor.tokenizer.padding_side = "right"
    processor.qformer_tokenizer.padding_side = "right"
    text_tensor = processor(
        text=final_text + processor.tokenizer.eos_token,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    instruction = processor(
        text=prompt,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_mask = torch.from_numpy(np.ascontiguousarray(input_mask.copy()))
    input_context_mask = get_context_mask(input_mask, input_size=input_size, enlarge_ratio=context_enlarge_ratio)

    return (
        image, input_mask, input_context_mask, text_tensor["input_ids"], text_tensor["attention_mask"],
         instruction["qformer_input_ids"], instruction["qformer_attention_mask"])

def get_all_a847_class():
        # Note that trainId is used!
        a847_meta = openseg_classes.get_ade20k_847_categories_with_prompt_eng()
        if LOWER_CHAR:
            a847_class = {x["trainId"]: {"name": x["name"].lower().split(","), "isthing": 0} for x in a847_meta}
        else:
            a847_class = {x["trainId"]: {"name": x["name"].split(","), "isthing": 0} for x in a847_meta}
        return a847_class

all_a847_class = get_all_a847_class()

def preprocess_a847(
    sample,
    processor,
    max_tokens,
    input_size,
    random_scale_min,
    random_scale_max,
    context_enlarge_ratio
):
    
    def a847_id2class(idx, all_a847_class):
        return all_a847_class[idx]["name"]


    return parse_sample_a847(sample, processor, max_tokens,
                              id2class_func=functools.partial(a847_id2class, all_a847_class=all_a847_class),
                              dataset_name="A847 Semantic", ignore_value_list=[65535,],
                              input_size=input_size, random_scale_min=random_scale_min,
                              random_scale_max=random_scale_max, context_enlarge_ratio=context_enlarge_ratio)

def get_all_pc459_class():
        pc459_meta = openseg_classes.get_pascal_ctx_459_categories_with_prompt_eng()
        if LOWER_CHAR:
            pc459_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": 0} for x in pc459_meta}
        else:
            pc459_class = {x["id"]: {"name": x["name"].split(","), "isthing": 0} for x in pc459_meta}
        return pc459_class

all_pc459_class = get_all_pc459_class()

def preprocess_pc459(
    sample,
    processor,
    max_tokens,
    input_size,
    random_scale_min,
    random_scale_max,
    context_enlarge_ratio
):
    
    def pc459_id2class(idx, all_pc459_class):
        return all_pc459_class[idx]["name"]

    return parse_sample_a847(sample, processor, max_tokens,
                              id2class_func=functools.partial(pc459_id2class, all_pc459_class=all_pc459_class),
                              dataset_name="PC459 Semantic", ignore_value_list=[65535, 430],
                              input_size=input_size, random_scale_min=random_scale_min,
                              random_scale_max=random_scale_max, context_enlarge_ratio=context_enlarge_ratio)

def parse_partimagenet(sample, processor, max_tokens, id2class_func, id2superclass_func, dataset_name,
                       input_size, random_scale_min, random_scale_max, context_enlarge_ratio, part_in_use_whole_prob, part_in_prepend_object_class_prob):
    info = json.loads(sample[0])
    image = info["image_base64"]
    rawbytes = base64.b64decode(image)
    image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
    prompt = random.choice(MASK_PROMPT)

    mode = "part"
    valid_ids = list(range(40))
    if random.random() < part_in_use_whole_prob:
        mode = "whole"
        valid_ids = list(range(158))

    sem_seg = info[f"{mode}_seg_base64"]
    rawbytes = base64.b64decode(sem_seg)
    sem_seg = Image.open(io.BytesIO(rawbytes))
    map_pretrain_object_class_for_part = (random.random() < part_in_prepend_object_class_prob) and (mode == "part")
    if map_pretrain_object_class_for_part:
        object_seg = info[f"whole_seg_base64"]
        rawbytes = base64.b64decode(object_seg)
        object_seg = Image.open(io.BytesIO(rawbytes))
        object_seg = np.array(object_seg)
        object_classes = None
        for obj_unique_id in np.unique(object_seg):
            if obj_unique_id not in list(range(158)):
                continue
            assert object_classes is None
            object_classes = id2class_func(obj_unique_id, mode='whole')[0]

    scale_factor = np.random.uniform(random_scale_min, random_scale_max)
    if min(image.size) < input_size:
        target_size = int(scale_factor * input_size)
        if min(image.size) == max(image.size):
            image = T.functional.resize(image, size=target_size,
                                    interpolation=T.functional.InterpolationMode.BICUBIC)
            sem_seg = T.functional.resize(sem_seg, size=target_size,
                                    interpolation=T.functional.InterpolationMode.NEAREST)
        else:
            image = T.functional.resize(image, size=target_size-1, max_size=target_size,
                                        interpolation=T.functional.InterpolationMode.BICUBIC)
            sem_seg = T.functional.resize(sem_seg, size=target_size-1, max_size=target_size,
                                    interpolation=T.functional.InterpolationMode.NEAREST)

    image = np.array(image)
    sem_seg = np.array(sem_seg) # H x W

    # random flip
    if torch.rand(1) < 0.5:
        image = image[:, ::-1]
        sem_seg = sem_seg[:, ::-1]

    try_cnt = 0
    while True:
        random_h = np.random.randint(0, max(0, sem_seg.shape[0] - input_size) + 1)
        random_w = np.random.randint(0, max(0, sem_seg.shape[1] - input_size) + 1)
        image_ = image[random_h:random_h+input_size, random_w:random_w+input_size, :].copy()
        sem_seg_ = sem_seg[random_h:random_h+input_size, random_w:random_w+input_size].copy()
        # padding to input_size
        padded_image = np.zeros(shape=(input_size, input_size, 3), dtype=np.uint8)
        padded_sem_seg = -np.ones(shape=(input_size, input_size), dtype=np.int32)
        padded_image[:image_.shape[0], :image_.shape[1]] = image_
        padded_sem_seg[:sem_seg_.shape[0], :sem_seg_.shape[1]] = sem_seg_

        image_ = padded_image
        sem_seg_ = padded_sem_seg

        # next, we generate masks
        masks = []
        classes = []

        all_ids = np.unique(sem_seg_)
        for unique_id in all_ids:
            if unique_id not in valid_ids:
                continue
            binary_mask = (sem_seg_ == unique_id)
            if binary_mask.sum() <= MIN_MASK_SIZE:
                continue
            masks.append(binary_mask)
            class_name = id2class_func(unique_id, mode)[0]

            if map_pretrain_object_class_for_part:
                supercat = id2superclass_func(unique_id)
                class_name = class_name.replace(supercat, object_classes)
            # TODO: Should we remove the supercat name in all cases?
            classes.append(class_name)
        
        if len(masks) > 0:
            sem_seg = sem_seg_.copy()
            image = image_.copy()
            break
        try_cnt += 1
        if try_cnt >= MAX_RETRY_CNT:
            raise ValueError(f"{dataset_name}: try_cnt larger than {MAX_RETRY_CNT}")
    image = Image.fromarray(image)
    processor.image_processor.size = {"height": input_size, "width": input_size}
    image = processor(images=image, return_tensors="pt")["pixel_values"]
    # shuffle masks and classes
    all_idx = list(range(len(masks)))
    random.shuffle(all_idx)
    # We just keep the first mask + class
    input_mask = masks[all_idx[0]].reshape(1, input_size, input_size)
    target_class = classes[all_idx[0]].strip()

    final_text = prompt + ' ' + target_class
    processor.tokenizer.padding_side = "right"
    processor.qformer_tokenizer.padding_side = "right"
    text_tensor = processor(
        text=final_text + processor.tokenizer.eos_token,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    instruction = processor(
        text=prompt,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_mask = torch.from_numpy(np.ascontiguousarray(input_mask.copy()))
    input_context_mask = get_context_mask(input_mask, input_size=input_size, enlarge_ratio=context_enlarge_ratio)
    return (
        image, input_mask, input_context_mask, text_tensor["input_ids"], text_tensor["attention_mask"],
         instruction["qformer_input_ids"], instruction["qformer_attention_mask"])

def get_all_partimagenet_class():
        part_meta, whole_meta = openseg_classes.get_all_partimagenet_class()
        if LOWER_CHAR:
            part_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": 0, "supercategory": x["supercategory"].lower()} for x in part_meta}
            whole_class = {x["id"]: {"name": x["name"].lower().split(","), "isthing": 0} for x in whole_meta}
        else:
            part_class = {x["id"]: {"name": x["name"].split(","), "isthing": 0, "supercategory": x["supercategory"]} for x in part_meta}
            whole_class = {x["id"]: {"name": x["name"].split(","), "isthing": 0} for x in whole_meta}
        return {"part": part_class, "whole": whole_class}

partimagenet_class = get_all_partimagenet_class()

def preprocess_partimagenet(
    sample,
    processor,
    max_tokens,
    input_size,
    random_scale_min,
    random_scale_max,
    context_enlarge_ratio,
    part_in_use_whole_prob,
    part_in_prepend_object_class_prob
):
    
    def partimagenet_id2class(idx, mode, partimagenet_class):
        return partimagenet_class[mode][idx]["name"]

    def partimagenet_id2supercat(idx, partimagenet_class):
        return partimagenet_class["part"][idx]["supercategory"]



    return parse_partimagenet(sample, processor, max_tokens,
                              id2class_func=functools.partial(partimagenet_id2class, partimagenet_class=partimagenet_class),
                              id2superclass_func=functools.partial(partimagenet_id2supercat, partimagenet_class=partimagenet_class),
                              dataset_name="PartImageNet Semantic", input_size=input_size,
                              random_scale_min=random_scale_min, random_scale_max=random_scale_max,
                              context_enlarge_ratio=context_enlarge_ratio, part_in_use_whole_prob=part_in_use_whole_prob,
                              part_in_prepend_object_class_prob=part_in_prepend_object_class_prob)

def parse_pascal_part(sample, processor, max_tokens, id2class_func, dataset_name, valid_ids,
                      input_size, random_scale_min, random_scale_max, context_enlarge_ratio):
    info = json.loads(sample[0])
    image = info["image_base64"]
    rawbytes = base64.b64decode(image)
    image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
    prompt = random.choice(MASK_PROMPT)

    mode = "part"
    sem_seg = info[f"{mode}_seg_base64"]
    rawbytes = base64.b64decode(sem_seg)
    sem_seg = Image.open(io.BytesIO(rawbytes))

    scale_factor = np.random.uniform(random_scale_min, random_scale_max)
    if min(image.size) < input_size:
        target_size = int(scale_factor * input_size)
        if min(image.size) == max(image.size):
            image = T.functional.resize(image, size=target_size,
                                    interpolation=T.functional.InterpolationMode.BICUBIC)
            sem_seg = T.functional.resize(sem_seg, size=target_size,
                                    interpolation=T.functional.InterpolationMode.NEAREST)
        else:
            image = T.functional.resize(image, size=target_size-1, max_size=target_size,
                                        interpolation=T.functional.InterpolationMode.BICUBIC)
            sem_seg = T.functional.resize(sem_seg, size=target_size-1, max_size=target_size,
                                    interpolation=T.functional.InterpolationMode.NEAREST)

    image = np.array(image)
    sem_seg = rgb2id(np.array(sem_seg)) # H x W

    # random flip
    if torch.rand(1) < 0.5:
        image = image[:, ::-1]
        sem_seg = sem_seg[:, ::-1]

    try_cnt = 0
    while True:
        random_h = np.random.randint(0, max(0, sem_seg.shape[0] - input_size) + 1)
        random_w = np.random.randint(0, max(0, sem_seg.shape[1] - input_size) + 1)
        image_ = image[random_h:random_h+input_size, random_w:random_w+input_size, :].copy()
        sem_seg_ = sem_seg[random_h:random_h+input_size, random_w:random_w+input_size].copy()
        # padding to input_size
        padded_image = np.zeros(shape=(input_size, input_size, 3), dtype=np.uint8)
        padded_sem_seg = -np.ones(shape=(input_size, input_size), dtype=np.int32)
        padded_image[:image_.shape[0], :image_.shape[1]] = image_
        padded_sem_seg[:sem_seg_.shape[0], :sem_seg_.shape[1]] = sem_seg_

        image_ = padded_image
        sem_seg_ = padded_sem_seg

        # next, we generate masks
        masks = []
        classes = []

        all_ids = np.unique(sem_seg_)
        for unique_id in all_ids:
            if unique_id not in valid_ids:
                continue
            binary_mask = (sem_seg_ == unique_id)
            if binary_mask.sum() <= MIN_MASK_SIZE:
                continue
            masks.append(binary_mask)
            class_name = id2class_func(unique_id)[0]
            classes.append(class_name)
        
        if len(masks) > 0:
            sem_seg = sem_seg_.copy()
            image = image_.copy()
            break
        try_cnt += 1
        if try_cnt >= MAX_RETRY_CNT:
            raise ValueError(f"{dataset_name}: try_cnt larger than {MAX_RETRY_CNT}")
    image = Image.fromarray(image)
    processor.image_processor.size = {"height": input_size, "width": input_size}
    image = processor(images=image, return_tensors="pt")["pixel_values"]
    # shuffle masks and classes
    all_idx = list(range(len(masks)))
    random.shuffle(all_idx)
    # We just keep the first mask + class
    input_mask = masks[all_idx[0]].reshape(1, input_size, input_size)
    target_class = classes[all_idx[0]].strip()

    final_text = prompt + ' ' + target_class
    processor.tokenizer.padding_side = "right"
    processor.qformer_tokenizer.padding_side = "right"
    text_tensor = processor(
        text=final_text + processor.tokenizer.eos_token,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    instruction = processor(
        text=prompt,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_mask = torch.from_numpy(np.ascontiguousarray(input_mask.copy()))
    input_context_mask = get_context_mask(input_mask, input_size=input_size, enlarge_ratio=context_enlarge_ratio)
    return (
        image, input_mask, input_context_mask, text_tensor["input_ids"], text_tensor["attention_mask"],
         instruction["qformer_input_ids"], instruction["qformer_attention_mask"])

def get_all_pascal_part_class():
        pascal_part_meta = openseg_classes.get_all_pascal_part_class()
        if LOWER_CHAR:
            for k in pascal_part_meta:
                pascal_part_meta[k] = {"name": pascal_part_meta[k].lower().split(","), "isthing": 0}
        else:
            for k in pascal_part_meta:
                pascal_part_meta[k] = {"name": pascal_part_meta[k].split(","), "isthing": 0}
        return pascal_part_meta

pascal_part_class = get_all_pascal_part_class()

def preprocess_pascal_part(
    sample,
    processor,
    max_tokens,
    input_size,
    random_scale_min,
    random_scale_max,
    context_enlarge_ratio
):
    
    def pascal_part_id2class(idx, pascal_part_class):
        return pascal_part_class[idx]["name"]

    return parse_pascal_part(sample, processor, max_tokens,
                              id2class_func=functools.partial(pascal_part_id2class, pascal_part_class=pascal_part_class),
                              dataset_name="PartImageNet Semantic",
                              valid_ids=pascal_part_class.keys(),
                              input_size=input_size, random_scale_min=random_scale_min,
                              random_scale_max=random_scale_max, context_enlarge_ratio=context_enlarge_ratio)
