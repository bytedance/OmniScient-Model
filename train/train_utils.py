"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/train_utils.py
"""

from contextlib import suppress
import torch
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
import os

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress


def train_one_epoch(
    args,
    model,
    epoch,
    coco_loader,
    lvis_loader,
    v3det_loader,
    a847_loader,
    pc459_loader,
    partimagenet_loader,
    pascal_part_loader,
    ade20k_loader,
    cityscapes_loader,
    processor,
    optimizer,
    lr_scheduler,
    device_id,
):
    all_loader = []
    prev_num_batches = None
    # setup loaders
    num_batches_per_epoch = None
    if coco_loader is not None:
        all_loader.append(coco_loader)
        num_batches_per_epoch_coco = coco_loader.num_batches
        if prev_num_batches is None:
            prev_num_batches = num_batches_per_epoch_coco
        else:
            assert (num_batches_per_epoch_coco == prev_num_batches
                    ), "Number of batches in all datasets must be the same"
        num_batches_per_epoch = num_batches_per_epoch_coco
    if lvis_loader is not None:
        all_loader.append(lvis_loader)
        num_batches_per_epoch_lvis = lvis_loader.num_batches
        if prev_num_batches is None:
            prev_num_batches = num_batches_per_epoch_lvis
        else:
            assert (num_batches_per_epoch_lvis == prev_num_batches
                    ), "Number of batches in all datasets must be the same"
        if num_batches_per_epoch is None:
            num_batches_per_epoch = num_batches_per_epoch_lvis
    if v3det_loader is not None:
        all_loader.append(v3det_loader)
        num_batches_per_epoch_v3det = v3det_loader.num_batches
        if prev_num_batches is None:
            prev_num_batches = num_batches_per_epoch_v3det
        else:
            assert (num_batches_per_epoch_v3det == prev_num_batches
                    ), "Number of batches in all datasets must be the same"
        if num_batches_per_epoch is None:
            num_batches_per_epoch = num_batches_per_epoch_v3det
    if a847_loader is not None:
        all_loader.append(a847_loader)
        num_batches_per_epoch_a847 = a847_loader.num_batches
        if prev_num_batches is None:
            prev_num_batches = num_batches_per_epoch_a847
        else:
            assert (num_batches_per_epoch_a847 == prev_num_batches
                    ), "Number of batches in all datasets must be the same"
        if num_batches_per_epoch is None:
            num_batches_per_epoch = num_batches_per_epoch_a847
    if pc459_loader is not None:
        all_loader.append(pc459_loader)
        num_batches_per_epoch_pc459 = pc459_loader.num_batches
        if prev_num_batches is None:
            prev_num_batches = num_batches_per_epoch_pc459
        else:
            assert (num_batches_per_epoch_pc459 == prev_num_batches
                    ), "Number of batches in all datasets must be the same"
        if num_batches_per_epoch is None:
            num_batches_per_epoch = num_batches_per_epoch_pc459
    if partimagenet_loader is not None:
        all_loader.append(partimagenet_loader)
        num_batches_per_epoch_partimagenet = partimagenet_loader.num_batches
        if prev_num_batches is None:
            prev_num_batches = num_batches_per_epoch_partimagenet
        else:
            assert (num_batches_per_epoch_partimagenet == prev_num_batches
                    ), "Number of batches in all datasets must be the same"
        if num_batches_per_epoch is None:
            num_batches_per_epoch = num_batches_per_epoch_partimagenet
    if pascal_part_loader is not None:
        all_loader.append(pascal_part_loader)
        num_batches_per_epoch_pascal_part = pascal_part_loader.num_batches
        if prev_num_batches is None:
            prev_num_batches = num_batches_per_epoch_pascal_part
        else:
            assert (num_batches_per_epoch_pascal_part == prev_num_batches
                    ), "Number of batches in all datasets must be the same"
        if num_batches_per_epoch is None:
            num_batches_per_epoch = num_batches_per_epoch_pascal_part
    if ade20k_loader is not None:
        all_loader.append(ade20k_loader)
        num_batches_per_epoch_ade20k = ade20k_loader.num_batches
        if prev_num_batches is None:
            prev_num_batches = num_batches_per_epoch_ade20k
        else:
            assert (num_batches_per_epoch_ade20k == prev_num_batches
                    ), "Number of batches in all datasets must be the same"
        if num_batches_per_epoch is None:
            num_batches_per_epoch = num_batches_per_epoch_ade20k
    
    if cityscapes_loader is not None:
        all_loader.append(cityscapes_loader)
        num_batches_per_epoch_cityscapes = cityscapes_loader.num_batches
        if prev_num_batches is None:
            prev_num_batches = num_batches_per_epoch_cityscapes
        else:
            assert (num_batches_per_epoch_cityscapes == prev_num_batches
                    ), "Number of batches in all datasets must be the same"
        if num_batches_per_epoch is None:
            num_batches_per_epoch = num_batches_per_epoch_cityscapes


    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    for num_steps, all_batch in tqdm(
        enumerate(zip(*all_loader)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):

        cur_idx = 0
        if coco_loader is not None:
            batch_coco = all_batch[cur_idx]
            cur_idx += 1
        if lvis_loader is not None:
            batch_lvis = all_batch[cur_idx]
            cur_idx += 1
        if v3det_loader is not None:
            batch_v3det = all_batch[cur_idx]
            cur_idx += 1
        if a847_loader is not None:
            batch_a847 = all_batch[cur_idx]
            cur_idx += 1
        if pc459_loader is not None:
            batch_pc459 = all_batch[cur_idx]
            cur_idx += 1
        if partimagenet_loader is not None:
            batch_partimagenet = all_batch[cur_idx]
            cur_idx += 1
        if pascal_part_loader is not None:
            batch_pascal_part = all_batch[cur_idx]
            cur_idx += 1
        if ade20k_loader is not None:
            batch_ade20k = all_batch[cur_idx]
            cur_idx += 1
        if cityscapes_loader is not None:
            batch_cityscapes = all_batch[cur_idx]
            cur_idx += 1


        def parse_data(batch_data):
            (images, masks, input_context_mask, input_ids,
              attention_mask, qformer_input_ids, qformer_attention_mask) = (
                batch_data[0], batch_data[1], batch_data[2], batch_data[3],
                  batch_data[4], batch_data[5], batch_data[6])
            images = images.to(device_id, dtype=cast_dtype, non_blocking=True).squeeze(1)
            masks = masks.to(device_id, dtype=cast_dtype, non_blocking=True)
            input_context_mask = input_context_mask.to(device_id, dtype=cast_dtype, non_blocking=True)
            input_ids = input_ids.to(device_id, dtype=None, non_blocking=True).squeeze(1)
            attention_mask = attention_mask.to(device_id, dtype=None, non_blocking=True).squeeze(1)
            qformer_input_ids = qformer_input_ids.to(device_id, dtype=None, non_blocking=True).squeeze(1)
            qformer_attention_mask = qformer_attention_mask.to(device_id, dtype=None, non_blocking=True).squeeze(1)
            # set up labels; language model is expected to handle shifting
            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels[:, 0] = -100
            labels = labels.to(device_id)

            return (images, masks, input_context_mask, input_ids,
                     attention_mask, qformer_input_ids, qformer_attention_mask, labels)

        if coco_loader is not None:
            #### COCO FORWARD PASS ####
            (images, masks, input_context_mask, input_ids,
                attention_mask, qformer_input_ids, qformer_attention_mask, labels) = parse_data(batch_coco)
            batch_size = images.shape[0]
            with autocast():
                loss_coco = (model(
                        pixel_values=images[:batch_size//2],
                        qformer_input_ids=qformer_input_ids[:batch_size//2],
                        qformer_attention_mask=qformer_attention_mask[:batch_size//2],
                        input_ids=input_ids[:batch_size//2],
                        attention_mask=attention_mask[:batch_size//2],
                        labels=labels[:batch_size//2],
                        segmentation_mask=masks[:batch_size//2],
                        input_context_mask=input_context_mask[:batch_size//2],
                        dataset_type="coco",
                    ) + model(
                        pixel_values=images[batch_size//2:],
                        qformer_input_ids=qformer_input_ids[batch_size//2:],
                        qformer_attention_mask=qformer_attention_mask[batch_size//2:],
                        input_ids=input_ids[batch_size//2:],
                        attention_mask=attention_mask[batch_size//2:],
                        labels=labels[batch_size//2:],
                        segmentation_mask=masks[batch_size//2:],
                        input_context_mask=input_context_mask[batch_size//2:],
                        dataset_type="any",
                    )) / 2.0
                # if loss is nan, skip this batch
                # this hack of skipping the batch is not FSDP-compatible
                if torch.isnan(loss_coco):
                    print("loss is nan, skipping this batch")
                    print("input_ids: ", processor.tokenizer.batch_decode(input_ids))
                    print("labels: ", labels)
                    print("images: ", images)
                    optimizer.zero_grad(set_to_none=True)
                    continue
            divided_loss_coco = loss_coco / args.gradient_accumulation_steps
            (divided_loss_coco * args.loss_multiplier_coco).backward()

        if lvis_loader is not None:
            #### LVIS FORWARD PASS ####
            (images, masks, input_context_mask, input_ids,
                attention_mask, qformer_input_ids, qformer_attention_mask, labels) = parse_data(batch_lvis)

            batch_size = images.shape[0]
            with autocast():
                loss_lvis = (model(
                        pixel_values=images[:batch_size//2],
                        qformer_input_ids=qformer_input_ids[:batch_size//2],
                        qformer_attention_mask=qformer_attention_mask[:batch_size//2],
                        input_ids=input_ids[:batch_size//2],
                        attention_mask=attention_mask[:batch_size//2],
                        labels=labels[:batch_size//2],
                        segmentation_mask=masks[:batch_size//2],
                        input_context_mask=input_context_mask[:batch_size//2],
                        dataset_type="lvis",
                    ) + model(
                        pixel_values=images[batch_size//2:],
                        qformer_input_ids=qformer_input_ids[batch_size//2:],
                        qformer_attention_mask=qformer_attention_mask[batch_size//2:],
                        input_ids=input_ids[batch_size//2:],
                        attention_mask=attention_mask[batch_size//2:],
                        labels=labels[batch_size//2:],
                        segmentation_mask=masks[batch_size//2:],
                        input_context_mask=input_context_mask[batch_size//2:],
                        dataset_type="any",
                    )) / 2.0
            
                # if loss is nan, skip this batch
                # this hack of skipping the batch is not FSDP-compatible
                if torch.isnan(loss_lvis):
                    print("loss is nan, skipping this batch")
                    print("input_ids: ", processor.tokenizer.batch_decode(input_ids))
                    print("labels: ", labels)
                    print("images: ", images)
                    optimizer.zero_grad(set_to_none=True)
                    continue

            divided_loss_lvis = loss_lvis / args.gradient_accumulation_steps
            (divided_loss_lvis * args.loss_multiplier_lvis).backward()

        if v3det_loader is not None:
            #### V3DET FORWARD PASS ####
            (images, masks, input_context_mask, input_ids,
                attention_mask, qformer_input_ids, qformer_attention_mask, labels) = parse_data(batch_v3det)

            batch_size = images.shape[0]
            with autocast():
                loss_v3det = (model(
                        pixel_values=images[:batch_size//2],
                        qformer_input_ids=qformer_input_ids[:batch_size//2],
                        qformer_attention_mask=qformer_attention_mask[:batch_size//2],
                        input_ids=input_ids[:batch_size//2],
                        attention_mask=attention_mask[:batch_size//2],
                        labels=labels[:batch_size//2],
                        segmentation_mask=masks[:batch_size//2],
                        input_context_mask=input_context_mask[:batch_size//2],
                        dataset_type="v3det",
                    ) + model(
                        pixel_values=images[batch_size//2:],
                        qformer_input_ids=qformer_input_ids[batch_size//2:],
                        qformer_attention_mask=qformer_attention_mask[batch_size//2:],
                        input_ids=input_ids[batch_size//2:],
                        attention_mask=attention_mask[batch_size//2:],
                        labels=labels[batch_size//2:],
                        segmentation_mask=masks[batch_size//2:],
                        input_context_mask=input_context_mask[batch_size//2:],
                        dataset_type="any",
                    )) / 2.0

                # if loss is nan, skip this batch
                # this hack of skipping the batch is not FSDP-compatible
                if torch.isnan(loss_v3det):
                    print("loss is nan, skipping this batch")
                    print("input_ids: ", processor.tokenizer.batch_decode(input_ids))
                    print("labels: ", labels)
                    print("images: ", images)
                    optimizer.zero_grad(set_to_none=True)
                    continue

            divided_loss_v3det = loss_v3det / args.gradient_accumulation_steps
            (divided_loss_v3det * args.loss_multiplier_v3det).backward()
        
        if a847_loader is not None:
            #### A847 FORWARD PASS ####
            (images, masks, input_context_mask, input_ids,
                attention_mask, qformer_input_ids, qformer_attention_mask, labels)= parse_data(batch_a847)

            batch_size = images.shape[0]
            with autocast():
                loss_a847 = (model(
                        pixel_values=images[:batch_size//2],
                        qformer_input_ids=qformer_input_ids[:batch_size//2],
                        qformer_attention_mask=qformer_attention_mask[:batch_size//2],
                        input_ids=input_ids[:batch_size//2],
                        attention_mask=attention_mask[:batch_size//2],
                        labels=labels[:batch_size//2],
                        segmentation_mask=masks[:batch_size//2],
                        input_context_mask=input_context_mask[:batch_size//2],
                        dataset_type="a847",
                    ) + model(
                        pixel_values=images[batch_size//2:],
                        qformer_input_ids=qformer_input_ids[batch_size//2:],
                        qformer_attention_mask=qformer_attention_mask[batch_size//2:],
                        input_ids=input_ids[batch_size//2:],
                        attention_mask=attention_mask[batch_size//2:],
                        labels=labels[batch_size//2:],
                        segmentation_mask=masks[batch_size//2:],
                        input_context_mask=input_context_mask[batch_size//2:],
                        dataset_type="any",
                    )) / 2.0
            
                # if loss is nan, skip this batch
                # this hack of skipping the batch is not FSDP-compatible
                if torch.isnan(loss_a847):
                    print("loss is nan, skipping this batch")
                    print("input_ids: ", processor.tokenizer.batch_decode(input_ids))
                    print("labels: ", labels)
                    print("images: ", images)
                    optimizer.zero_grad(set_to_none=True)
                    continue

            divided_loss_a847 = loss_a847 / args.gradient_accumulation_steps
            (divided_loss_a847 * args.loss_multiplier_a847).backward()

        if pc459_loader is not None:
            #### PC459 FORWARD PASS ####
            (images, masks, input_context_mask, input_ids,
                attention_mask, qformer_input_ids, qformer_attention_mask, labels) = parse_data(batch_pc459)

            batch_size = images.shape[0]
            with autocast():
                loss_pc459 = (model(
                        pixel_values=images[:batch_size//2],
                        qformer_input_ids=qformer_input_ids[:batch_size//2],
                        qformer_attention_mask=qformer_attention_mask[:batch_size//2],
                        input_ids=input_ids[:batch_size//2],
                        attention_mask=attention_mask[:batch_size//2],
                        labels=labels[:batch_size//2],
                        segmentation_mask=masks[:batch_size//2],
                        input_context_mask=input_context_mask[:batch_size//2],
                        dataset_type="pc459",
                    ) + model(
                        pixel_values=images[batch_size//2:],
                        qformer_input_ids=qformer_input_ids[batch_size//2:],
                        qformer_attention_mask=qformer_attention_mask[batch_size//2:],
                        input_ids=input_ids[batch_size//2:],
                        attention_mask=attention_mask[batch_size//2:],
                        labels=labels[batch_size//2:],
                        segmentation_mask=masks[batch_size//2:],
                        input_context_mask=input_context_mask[batch_size//2:],
                        dataset_type="any",
                    )) / 2.0

                # if loss is nan, skip this batch
                # this hack of skipping the batch is not FSDP-compatible
                if torch.isnan(loss_pc459):
                    print("loss is nan, skipping this batch")
                    print("input_ids: ", processor.tokenizer.batch_decode(input_ids))
                    print("labels: ", labels)
                    print("images: ", images)
                    optimizer.zero_grad(set_to_none=True)
                    continue

            divided_loss_pc459 = loss_pc459 / args.gradient_accumulation_steps
            (divided_loss_pc459 * args.loss_multiplier_pc459).backward()

        if partimagenet_loader is not None:
            #### PART-IMAGENET FORWARD PASS ####
            (images, masks, input_context_mask, input_ids,
                attention_mask, qformer_input_ids, qformer_attention_mask, labels) = parse_data(batch_partimagenet)

            batch_size = images.shape[0]
            with autocast():   
                loss_partimagenet = (model(
                        pixel_values=images[:batch_size//2],
                        qformer_input_ids=qformer_input_ids[:batch_size//2],
                        qformer_attention_mask=qformer_attention_mask[:batch_size//2],
                        input_ids=input_ids[:batch_size//2],
                        attention_mask=attention_mask[:batch_size//2],
                        labels=labels[:batch_size//2],
                        segmentation_mask=masks[:batch_size//2],
                        input_context_mask=input_context_mask[:batch_size//2],
                        dataset_type="partimagenet",
                    ) + model(
                        pixel_values=images[batch_size//2:],
                        qformer_input_ids=qformer_input_ids[batch_size//2:],
                        qformer_attention_mask=qformer_attention_mask[batch_size//2:],
                        input_ids=input_ids[batch_size//2:],
                        attention_mask=attention_mask[batch_size//2:],
                        labels=labels[batch_size//2:],
                        segmentation_mask=masks[batch_size//2:],
                        input_context_mask=input_context_mask[batch_size//2:],
                        dataset_type="any",
                    )) / 2.0

                # if loss is nan, skip this batch
                # this hack of skipping the batch is not FSDP-compatible
                if torch.isnan(loss_partimagenet):
                    print("loss is nan, skipping this batch")
                    print("input_ids: ", processor.tokenizer.batch_decode(input_ids))
                    print("labels: ", labels)
                    print("images: ", images)
                    optimizer.zero_grad(set_to_none=True)
                    continue

            divided_loss_partimagenet = loss_partimagenet / args.gradient_accumulation_steps
            (divided_loss_partimagenet * args.loss_multiplier_partimagenet).backward()

        if pascal_part_loader is not None:
            #### PASCAL PART FORWARD PASS ####
            (images, masks, input_context_mask, input_ids,
                attention_mask, qformer_input_ids, qformer_attention_mask, labels) = parse_data(batch_pascal_part)

            batch_size = images.shape[0]
            with autocast():
                loss_pascal_part = (model(
                        pixel_values=images[:batch_size//2],
                        qformer_input_ids=qformer_input_ids[:batch_size//2],
                        qformer_attention_mask=qformer_attention_mask[:batch_size//2],
                        input_ids=input_ids[:batch_size//2],
                        attention_mask=attention_mask[:batch_size//2],
                        labels=labels[:batch_size//2],
                        segmentation_mask=masks[:batch_size//2],
                        input_context_mask=input_context_mask[:batch_size//2],
                        dataset_type="pascal_part",
                    ) + model(
                        pixel_values=images[batch_size//2:],
                        qformer_input_ids=qformer_input_ids[batch_size//2:],
                        qformer_attention_mask=qformer_attention_mask[batch_size//2:],
                        input_ids=input_ids[batch_size//2:],
                        attention_mask=attention_mask[batch_size//2:],
                        labels=labels[batch_size//2:],
                        segmentation_mask=masks[batch_size//2:],
                        input_context_mask=input_context_mask[batch_size//2:],
                        dataset_type="any",
                    )) / 2.0

                # if loss is nan, skip this batch
                # this hack of skipping the batch is not FSDP-compatible
                if torch.isnan(loss_pascal_part):
                    print("loss is nan, skipping this batch")
                    print("input_ids: ", processor.tokenizer.batch_decode(input_ids))
                    print("labels: ", labels)
                    print("images: ", images)
                    optimizer.zero_grad(set_to_none=True)
                    continue

            divided_loss_pascal_part = loss_pascal_part / args.gradient_accumulation_steps
            (divided_loss_pascal_part * args.loss_multiplier_pascal_part).backward()

        if ade20k_loader is not None:
            #### ADE20K FORWARD PASS ####
            (images, masks, input_context_mask, input_ids,
                attention_mask, qformer_input_ids, qformer_attention_mask, labels) = parse_data(batch_ade20k)
            
            batch_size = images.shape[0]
            with autocast():
                loss_ade20k = (model(
                        pixel_values=images[:batch_size//2],
                        qformer_input_ids=qformer_input_ids[:batch_size//2],
                        qformer_attention_mask=qformer_attention_mask[:batch_size//2],
                        input_ids=input_ids[:batch_size//2],
                        attention_mask=attention_mask[:batch_size//2],
                        labels=labels[:batch_size//2],
                        segmentation_mask=masks[:batch_size//2],
                        input_context_mask=input_context_mask[:batch_size//2],
                        dataset_type="ade20k",
                    ) + model(
                        pixel_values=images[batch_size//2:],
                        qformer_input_ids=qformer_input_ids[batch_size//2:],
                        qformer_attention_mask=qformer_attention_mask[batch_size//2:],
                        input_ids=input_ids[batch_size//2:],
                        attention_mask=attention_mask[batch_size//2:],
                        labels=labels[batch_size//2:],
                        segmentation_mask=masks[batch_size//2:],
                        input_context_mask=input_context_mask[batch_size//2:],
                        dataset_type="any",
                    )) / 2.0

                # if loss is nan, skip this batch
                # this hack of skipping the batch is not FSDP-compatible
                if torch.isnan(loss_ade20k):
                    print("loss is nan, skipping this batch")
                    print("input_ids: ", processor.tokenizer.batch_decode(input_ids))
                    print("labels: ", labels)
                    print("images: ", images)
                    optimizer.zero_grad(set_to_none=True)
                    continue
            divided_loss_ade20k = loss_ade20k / args.gradient_accumulation_steps
            (divided_loss_ade20k * args.loss_multiplier_ade20k).backward()

        if cityscapes_loader is not None:
            #### CityScapes FORWARD PASS ####
            (images, masks, input_context_mask, input_ids,
                attention_mask, qformer_input_ids, qformer_attention_mask, labels) = parse_data(batch_cityscapes)
            
            batch_size = images.shape[0]
            with autocast():                    
                loss_cityscapes = (model(
                        pixel_values=images[:batch_size//2],
                        qformer_input_ids=qformer_input_ids[:batch_size//2],
                        qformer_attention_mask=qformer_attention_mask[:batch_size//2],
                        input_ids=input_ids[:batch_size//2],
                        attention_mask=attention_mask[:batch_size//2],
                        labels=labels[:batch_size//2],
                        segmentation_mask=masks[:batch_size//2],
                        input_context_mask=input_context_mask[:batch_size//2],
                        dataset_type="cityscapes",
                    ) + model(
                        pixel_values=images[batch_size//2:],
                        qformer_input_ids=qformer_input_ids[batch_size//2:],
                        qformer_attention_mask=qformer_attention_mask[batch_size//2:],
                        input_ids=input_ids[batch_size//2:],
                        attention_mask=attention_mask[batch_size//2:],
                        labels=labels[batch_size//2:],
                        segmentation_mask=masks[batch_size//2:],
                        input_context_mask=input_context_mask[batch_size//2:],
                        dataset_type="any",
                    )) / 2.0

                # if loss is nan, skip this batch
                # this hack of skipping the batch is not FSDP-compatible
                if torch.isnan(loss_cityscapes):
                    print("loss is nan, skipping this batch")
                    print("input_ids: ", processor.tokenizer.batch_decode(input_ids))
                    print("labels: ", labels)
                    print("images: ", images)
                    optimizer.zero_grad(set_to_none=True)
                    continue
            divided_loss_cityscapes = loss_cityscapes / args.gradient_accumulation_steps
            (divided_loss_cityscapes * args.loss_multiplier_cityscapes).backward()

        if args.rank == 0:
            logging_content = f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete."
            if coco_loader is not None:
                logging_content += f" Loss COCO: {loss_coco.item():.3f}"
            if lvis_loader is not None:
                logging_content += f" Loss LVIS: {loss_lvis.item():.3f}"
            if a847_loader is not None:
                logging_content += f" Loss A847: {loss_a847.item():.3f}"
            if pc459_loader is not None:
                logging_content += f" Loss PC459: {loss_pc459.item():.3f}"
            if partimagenet_loader is not None:
                logging_content += f" Loss PART-IMAGENET: {loss_partimagenet.item():.3f}"
            if pascal_part_loader is not None:
                logging_content += f" Loss PascalPart: {loss_pascal_part.item():.3f}"
            if v3det_loader is not None:
                logging_content += f" Loss V3DET: {loss_v3det.item():.3f}"
            if ade20k_loader is not None:
                logging_content += f" Loss ADE20K: {loss_ade20k.item():.3f}"
            if cityscapes_loader is not None:
                logging_content += f" Loss CityScapes: {loss_cityscapes.item():.3f}"
            print(logging_content)

        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            logging_content = f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete."
            if coco_loader is not None:
                logging_content += f" Loss COCO: {loss_coco.item():.3f}"
            if lvis_loader is not None:
                logging_content += f" Loss LVIS: {loss_lvis.item():.3f}"
            if a847_loader is not None:
                logging_content += f" Loss A847: {loss_a847.item():.3f}"
            if pc459_loader is not None:
                logging_content += f" Loss PC459: {loss_pc459.item():.3f}"
            if partimagenet_loader is not None:
                logging_content += f" Loss PART-IMAGENET: {loss_partimagenet.item():.3f}"
            if pascal_part_loader is not None:
                logging_content += f" Loss PascalPart: {loss_pascal_part.item():.3f}"
            if v3det_loader is not None:
                logging_content += f" Loss V3DET: {loss_v3det.item():.3f}"
            if ade20k_loader is not None:
                logging_content += f" Loss ADE20K: {loss_ade20k.item():.3f}"
            if cityscapes_loader is not None:
                logging_content += f" Loss CityScapes: {loss_cityscapes.item():.3f}"
            print(logging_content)


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    return state_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer, group=args.my_group)

    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        if not (args.fsdp and not args.fsdp_use_orig_params):
            model_state = filter_state_dict_to_trainable(model, model_state)

        run_name = args.run_name
        if not os.path.exists(run_name):
            os.makedirs(run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        print(f"Saving checkpoint to {run_name}/checkpoint_{epoch}.pt")
        torch.save(checkpoint_dict, f"{run_name}/checkpoint_{epoch}.pt")
        if args.delete_previous_checkpoint:
            if epoch > 0 and os.path.exists(f"{run_name}/checkpoint_{epoch-1}.pt"):
                os.remove(f"{run_name}/checkpoint_{epoch-1}.pt")
