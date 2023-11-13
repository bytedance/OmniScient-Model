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
import cv2
from utils import prepare_sam, prepare_osm, prepare_instruction, prepare_image, get_masks, get_context_mask, get_classes


@torch.no_grad()
def post_processing_to_panoptic_mask(masks, classes, class_probs, class_threshold=0.2,
                    overlapping_ratio=0.8, mask_min_size=100):
    assert len(masks) == len(classes) and len(classes) == len(class_probs)

    # post-processing as in kMaX-DeepLab style, to obtain non-overlapping masks (i.e., panoptic masks)
    class_probs = torch.tensor(class_probs)
    reorder_indices = torch.argsort(class_probs, dim=-1, descending=True)
    pan_mask = np.zeros_like(masks[0]).astype(np.uint8)
    final_classes = []
    new_idx = 1

    for i in range(len(masks)):
        cur_idx = reorder_indices[i].item() # 1
        cur_mask = masks[cur_idx]
        cur_class = classes[cur_idx]
        cur_prob = class_probs[cur_idx].item()
        if cur_prob < class_threshold:
            continue
        assert cur_class
        original_pixel_num = cur_mask.sum()
        new_binary_mask = np.logical_and(cur_mask, pan_mask==0)
        new_pixel_number = new_binary_mask.sum()
        if original_pixel_num * overlapping_ratio > new_pixel_number or new_pixel_number < mask_min_size:
            continue
        pan_mask[new_binary_mask] = new_idx
        final_classes.append(cur_class)
        new_idx += 1

    return pan_mask, final_classes, new_idx

@torch.no_grad()
def process_single_image(image_path, save_path, save_name, input_size, mask_generator,
                         class_generator, processor, qformer_lang_x, lang_x,
                         vis_alpha=0.7):
    image_for_osm, image_for_seg = prepare_image(image_path, input_size)

    # get mask
    seg_masks = get_masks(image_for_seg, mask_generator, input_size)
    if len(seg_masks) == 0:
        return
    
    # get class
    classes, class_probs = get_classes(image_for_osm, seg_masks, class_generator,
                                        processor, qformer_lang_x, lang_x)

    # post-process
    pan_mask, pan_classes, pan_max_idx = post_processing_to_panoptic_mask(
        seg_masks, classes, class_probs)
    
    # draw vis
    canvas = image_for_seg[:, :, ::-1] # RGB -> BGR
    # pan_idx 0 is reserved for void
    # firstly overlaying all masks
    for pan_idx in range(1, pan_max_idx):
        canvas = canvas.astype(np.float32)
        color = (np.random.random(3) * 255.0).astype(np.float32)
        cur_mask = (pan_mask == pan_idx).astype(np.float32)[:, :, None]
        canvas = (canvas * (1.0 - cur_mask) +
                  canvas * cur_mask * (1.0 - vis_alpha) +
                    color * cur_mask * vis_alpha)
    
    canvas = canvas.astype(np.uint8)

    # put all classes as text
    for pan_idx in range(1, pan_max_idx):
        cur_mask = (pan_mask == pan_idx)
        median = np.median(cur_mask.nonzero(), axis=1)[::-1] # hw to xy
        median = median.astype(np.int32)
        cur_class = pan_classes[pan_idx - 1]
        (w, h), _ = cv2.getTextSize(cur_class, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_x, text_y = median
        text_x = max(0, text_x - w//2)
        text_y = max(0, text_y)
        box_x1, box_y1 = text_x - 5, text_y - h - 5
        box_x2, box_y2 = text_x + w + 5, text_y + 5
        canvas = cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        canvas = cv2.putText(canvas, cur_class, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 246, 55), 1, cv2.LINE_AA)
        
    canvas = canvas[:, :, ::-1].astype(np.uint8) # BGR -> RGB
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print('saving ', os.path.join(save_path, save_name))
    Image.fromarray(canvas).save(os.path.join(save_path, save_name))
    return

@torch.no_grad()
def process_all_images(image_path_list, save_path, sam_checkpoint, osm_checkpoint):
    # prepare SAM model
    mask_generator, _ = prepare_sam(sam_checkpoint=sam_checkpoint)
    # prepare OSM model and instructions
    class_generator, processor = prepare_osm(osm_checkpoint=osm_checkpoint)
    lang_x, qformer_lang_x = prepare_instruction(
        processor, "What is in the segmentation mask? Assistant:")
    input_size = processor.image_processor.size["height"]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # process images one by one
    for image_path in image_path_list:
        file_name, file_ext = os.path.splitext(os.path.basename(image_path))
        save_name = file_name + '_vis.png'
        process_single_image(image_path, save_path, save_name,
                             input_size, mask_generator, 
                             class_generator, processor, qformer_lang_x, lang_x,
                             vis_alpha=0.7)
    return


if __name__ == "__main__":
    import glob
    image_list = glob.glob('imgs/*.jpg')
    save_path = "demo_vis"
    process_all_images(image_list, save_path, sam_checkpoint="./sam_vit_h_4b8939.pth", osm_checkpoint="./osm_final.pt")