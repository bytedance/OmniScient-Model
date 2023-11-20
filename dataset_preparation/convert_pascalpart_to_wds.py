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

import argparse
import json
import os
import uuid
import zipfile
from PIL import Image
import base64
from io import BytesIO
import webdataset as wds
import glob
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--output_dir",
    type=str,
    help="Pass in the directory where the output shards (as tar files) will be written to.",
)
arg_parser.add_argument(
    "--image_dir",
    type=str,
    help="Pass in the directory where the images have been downloaded to.",
)
arg_parser.add_argument(
    "--ann_dir",
    type=str,
)
arg_parser.add_argument(
    "--num_files_per_shard",
    type=int,
    default=1000,
)
args = arg_parser.parse_args()

def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color

import scipy.io

global_idx_name_mapping = {}
global_name_idx_mapping = {}
cur_idx = 1 # 0 for ignored
# ref https://github.com/micco00x/py-pascalpart/blob/master/utils.py#L4
def parse_mat(mat_file):
    global cur_idx
    assert cur_idx <= 255 * 255 * 255
    annotations = scipy.io.loadmat(mat_file)["anno"]
    objects = annotations[0, 0]["objects"]

    mapped_whole_mask = None
    mapped_part_mask = None

    # Go through the objects and extract info:
    for obj_idx in range(objects.shape[1]):
        obj = objects[0, obj_idx]

        # Get classname and mask of the current object:
        classname = obj["class"][0]

        if classname not in global_name_idx_mapping:
            global_name_idx_mapping[classname] = cur_idx
            global_idx_name_mapping[cur_idx] = classname
            cur_idx += 1

        mask = obj["mask"]
        if mapped_whole_mask is None:
            mapped_whole_mask = np.zeros_like(mask).astype(np.int32)
        mapped_whole_mask[mask == 1] = global_name_idx_mapping[classname]

        parts = obj["parts"]
        # Go through the part of the specific object and extract info:
        for part_idx in range(parts.shape[1]):
            part = parts[0, part_idx]
            # Get part name and mask of the current body part:
            part_name = part["part_name"][0]
            part_name = classname + ':' + part_name
            part_mask = part["mask"]

            if part_name not in global_name_idx_mapping:
                global_name_idx_mapping[part_name] = cur_idx
                global_idx_name_mapping[cur_idx] = part_name
                cur_idx += 1

            if mapped_part_mask is None:
                mapped_part_mask = np.zeros_like(part_mask).astype(np.int32)
            mapped_part_mask[part_mask == 1] = global_name_idx_mapping[part_name]

    if mapped_whole_mask is not None and mapped_part_mask is not None:
        mapped_whole_mask, mapped_part_mask = id2rgb(mapped_whole_mask), id2rgb(mapped_part_mask)
    return mapped_whole_mask, mapped_part_mask


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    all_labels = glob.glob(args.ann_dir + '/*.mat')
    import random
    random.seed(20230731)
    random.shuffle(all_labels)

    with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
        for idx in range(len(all_labels)):
            label_file = all_labels[idx]
            image_file = label_file.replace(args.ann_dir, args.image_dir).replace(".mat", ".jpg")
            sample_data = {}
            img = Image.open(image_file).convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            sample_data["image_base64"] = img_str.decode("utf-8")
            # convert to base64

            whole_seg, part_seg = parse_mat(label_file)

            if whole_seg is None or part_seg is None:
                continue
            part_seg = Image.fromarray(part_seg)
            buffered = BytesIO()
            part_seg.save(buffered, format="PNG")
            part_seg_str = base64.b64encode(buffered.getvalue())
            sample_data["part_seg_base64"] = part_seg_str.decode("utf-8")

            whole_seg = Image.fromarray(whole_seg)
            buffered = BytesIO()
            whole_seg.save(buffered, format="PNG")
            whole_seg_str = base64.b64encode(buffered.getvalue())
            sample_data["whole_seg_base64"] = whole_seg_str.decode("utf-8")

            key_str = uuid.uuid4().hex
            sink.write({"__key__": key_str, "json": sample_data})

            if (idx + 1) % args.num_files_per_shard == 0:
                sink.next_stream()
    print("global_idx_name_mapping:", global_idx_name_mapping)
    print("global_name_idx_mapping:", global_name_idx_mapping)

    with open("./pascal_part_class_mapping.txt", "w") as f:
        f.write("global_idx_name_mapping:" + str(global_idx_name_mapping) + '\n')
        f.write("global_name_idx_mapping:" + str(global_name_idx_mapping) + '\n')
    
"""
global_idx_name_mapping:{1: 'cow', 2: 'cow:head', 3: 'cow:lear', 4: 'cow:rear', 5: 'cow:torso', 6: 'cow:neck', 7: 'cow:lflleg', 8: 'cow:lfuleg', 9: 'cow:rfuleg', 10: 'cow:lbuleg', 11: 'cow:tail', 12: 'chair', 13: 'tvmonitor', 14: 'tvmonitor:screen', 15: 'person', 16: 'person:head', 17: 'person:torso', 18: 'person:llarm', 19: 'person:luarm', 20: 'person:llleg', 21: 'person:luleg', 22: 'car', 23: 'car:bliplate', 24: 'car:backside', 25: 'car:rightside', 26: 'car:rightmirror', 27: 'car:window_1', 28: 'car:window_2', 29: 'person:leye', 30: 'person:reye', 31: 'person:lebrow', 32: 'person:rebrow', 33: 'person:mouth', 34: 'person:hair', 35: 'person:nose', 36: 'person:neck', 37: 'person:rear', 38: 'person:lear', 39: 'person:ruarm', 40: 'person:lhand', 41: 'person:rlarm', 42: 'person:rhand', 43: 'person:rlleg', 44: 'person:ruleg', 45: 'person:lfoot', 46: 'person:rfoot', 47: 'motorbike', 48: 'motorbike:fwheel', 49: 'motorbike:bwheel', 50: 'car:frontside', 51: 'car:leftside', 52: 'car:wheel_1', 53: 'car:wheel_2', 54: 'bicycle', 55: 'bicycle:fwheel', 56: 'bicycle:bwheel', 57: 'bicycle:chainwheel', 58: 'bicycle:handlebar', 59: 'bicycle:saddle', 60: 'dog', 61: 'dog:head', 62: 'dog:rear', 63: 'dog:muzzle', 64: 'dog:nose', 65: 'dog:torso', 66: 'dog:neck', 67: 'dog:lfleg', 68: 'dog:lfpa', 69: 'dog:rfleg', 70: 'dog:rfpa', 71: 'dog:lbleg', 72: 'dog:lbpa', 73: 'dog:rbleg', 74: 'dog:rbpa', 75: 'dog:tail', 76: 'dog:lear', 77: 'dog:leye', 78: 'cat', 79: 'cat:head', 80: 'cat:lear', 81: 'cat:rear', 82: 'cat:leye', 83: 'cat:reye', 84: 'cat:nose', 85: 'cat:torso', 86: 'cat:neck', 87: 'cat:lfleg', 88: 'cat:lfpa', 89: 'cat:rfleg', 90: 'cat:rfpa', 91: 'cat:lbleg', 92: 'cat:lbpa', 93: 'cat:rbleg', 94: 'cat:rbpa', 95: 'cat:tail', 96: 'train', 97: 'train:head', 98: 'train:headlight_1', 99: 'train:hfrontside', 100: 'train:hleftside', 101: 'bird', 102: 'bird:head', 103: 'bird:reye', 104: 'bird:beak', 105: 'bird:torso', 106: 'bird:neck', 107: 'bird:lwing', 108: 'bird:rwing', 109: 'bird:lleg', 110: 'bird:lfoot', 111: 'bird:rleg', 112: 'bird:rfoot', 113: 'dog:reye', 114: 'train:coach_1', 115: 'train:cleftside_1', 116: 'train:headlight_2', 117: 'train:headlight_3', 118: 'horse', 119: 'horse:torso', 120: 'horse:lflleg', 121: 'horse:lfuleg', 122: 'horse:rflleg', 123: 'horse:rfuleg', 124: 'horse:head', 125: 'horse:lear', 126: 'horse:rear', 127: 'horse:leye', 128: 'horse:muzzle', 129: 'horse:neck', 130: 'horse:rblleg', 131: 'horse:rbuleg', 132: 'bottle', 133: 'bottle:body', 134: 'bottle:cap', 135: 'pottedplant', 136: 'pottedplant:pot', 137: 'pottedplant:plant', 138: 'car:door_1', 139: 'car:door_2', 140: 'train:crightside_1', 141: 'horse:reye', 142: 'horse:lfho', 143: 'horse:lblleg', 144: 'horse:lbuleg', 145: 'horse:tail', 146: 'horse:rfho', 147: 'horse:lbho', 148: 'car:leftmirror', 149: 'car:headlight_1', 150: 'car:headlight_2', 151: 'car:wheel_3', 152: 'car:wheel_4', 153: 'car:fliplate', 154: 'car:window_3', 155: 'bird:tail', 156: 'aeroplane', 157: 'aeroplane:body', 158: 'aeroplane:lwing', 159: 'aeroplane:stern', 160: 'aeroplane:rwing', 161: 'aeroplane:wheel_1', 162: 'aeroplane:wheel_2', 163: 'aeroplane:engine_1', 164: 'aeroplane:engine_2', 165: 'aeroplane:wheel_3', 166: 'aeroplane:tail', 167: 'table', 168: 'bird:leye', 169: 'train:hrightside', 170: 'cow:leye', 171: 'cow:muzzle', 172: 'cow:lblleg', 173: 'cow:rblleg', 174: 'cow:rbuleg', 175: 'boat', 176: 'motorbike:headlight_1', 177: 'sheep', 178: 'sheep:head', 179: 'sheep:torso', 180: 'sheep:neck', 181: 'sheep:tail', 182: 'sheep:muzzle', 183: 'sheep:lflleg', 184: 'sheep:lfuleg', 185: 'sheep:rflleg', 186: 'sheep:rfuleg', 187: 'sheep:lblleg', 188: 'sheep:rblleg', 189: 'sheep:rbuleg', 190: 'sheep:lear', 191: 'sheep:rear', 192: 'sheep:lbuleg', 193: 'bus', 194: 'bus:fliplate', 195: 'bus:frontside', 196: 'bus:rightside', 197: 'bus:leftmirror', 198: 'bus:rightmirror', 199: 'bus:headlight_1', 200: 'bus:headlight_2', 201: 'bus:headlight_3', 202: 'bus:headlight_4', 203: 'bus:wheel_1', 204: 'bus:wheel_2', 205: 'bus:window_1', 206: 'bus:window_2', 207: 'bus:window_3', 208: 'bus:window_4', 209: 'bus:window_5', 210: 'bus:window_6', 211: 'bus:window_7', 212: 'bus:window_8', 213: 'bus:window_9', 214: 'bus:window_10', 215: 'car:roofside', 216: 'sheep:reye', 217: 'bus:leftside', 218: 'bus:door_1', 219: 'bus:headlight_5', 220: 'bus:headlight_6', 221: 'sheep:leye', 222: 'car:window_4', 223: 'sofa', 224: 'bus:bliplate', 225: 'bus:backside', 226: 'bus:door_2', 227: 'bus:wheel_3', 228: 'train:coach_2', 229: 'train:coach_3', 230: 'train:coach_4', 231: 'train:coach_5', 232: 'train:coach_6', 233: 'train:coach_7', 234: 'train:coach_8', 235: 'train:cleftside_2', 236: 'train:cleftside_3', 237: 'train:cleftside_4', 238: 'train:cleftside_5', 239: 'train:cleftside_8', 240: 'train:crightside_6', 241: 'train:crightside_7', 242: 'train:crightside_2', 243: 'train:cbackside_1', 244: 'horse:rbho', 245: 'train:hroofside', 246: 'sheep:lhorn', 247: 'sheep:rhorn', 248: 'aeroplane:engine_3', 249: 'aeroplane:engine_4', 250: 'motorbike:headlight_2', 251: 'cow:rflleg', 252: 'cow:rhorn', 253: 'cow:lhorn', 254: 'aeroplane:wheel_4', 255: 'aeroplane:wheel_5', 256: 'aeroplane:wheel_6', 257: 'bus:window_11', 258: 'cow:reye', 259: 'bus:wheel_4', 260: 'train:croofside_1', 261: 'train:cfrontside_2', 262: 'aeroplane:wheel_7', 263: 'train:cfrontside_1', 264: 'train:headlight_4', 265: 'bus:roofside', 266: 'bus:headlight_7', 267: 'bus:headlight_8', 268: 'bus:window_12', 269: 'bus:window_13', 270: 'motorbike:handlebar', 271: 'motorbike:saddle', 272: 'bus:door_3', 273: 'car:window_5', 274: 'motorbike:headlight_3', 275: 'aeroplane:wheel_8', 276: 'car:headlight_3', 277: 'car:headlight_4', 278: 'car:headlight_5', 279: 'train:hbackside', 280: 'train:crightside_3', 281: 'train:croofside_2', 282: 'train:croofside_3', 283: 'bus:wheel_5', 284: 'car:window_6', 285: 'car:window_7', 286: 'train:cbackside_2', 287: 'car:headlight_6', 288: 'train:crightside_4', 289: 'train:crightside_5', 290: 'train:crightside_8', 291: 'train:coach_9', 292: 'train:cleftside_6', 293: 'train:cleftside_7', 294: 'train:cleftside_9', 295: 'train:cfrontside_3', 296: 'train:cfrontside_4', 297: 'train:cfrontside_5', 298: 'train:cfrontside_6', 299: 'train:cfrontside_7', 300: 'train:cfrontside_9', 301: 'train:headlight_5', 302: 'car:door_3', 303: 'bus:window_14', 304: 'train:croofside_4', 305: 'bus:window_15', 306: 'train:croofside_5', 307: 'car:wheel_5', 308: 'bicycle:headlight_1', 309: 'aeroplane:engine_5', 310: 'aeroplane:engine_6', 311: 'bus:window_16', 312: 'bus:window_17', 313: 'bus:window_18', 314: 'bus:window_19', 315: 'bus:door_4', 316: 'bus:window_20'}
global_name_idx_mapping:{'cow': 1, 'cow:head': 2, 'cow:lear': 3, 'cow:rear': 4, 'cow:torso': 5, 'cow:neck': 6, 'cow:lflleg': 7, 'cow:lfuleg': 8, 'cow:rfuleg': 9, 'cow:lbuleg': 10, 'cow:tail': 11, 'chair': 12, 'tvmonitor': 13, 'tvmonitor:screen': 14, 'person': 15, 'person:head': 16, 'person:torso': 17, 'person:llarm': 18, 'person:luarm': 19, 'person:llleg': 20, 'person:luleg': 21, 'car': 22, 'car:bliplate': 23, 'car:backside': 24, 'car:rightside': 25, 'car:rightmirror': 26, 'car:window_1': 27, 'car:window_2': 28, 'person:leye': 29, 'person:reye': 30, 'person:lebrow': 31, 'person:rebrow': 32, 'person:mouth': 33, 'person:hair': 34, 'person:nose': 35, 'person:neck': 36, 'person:rear': 37, 'person:lear': 38, 'person:ruarm': 39, 'person:lhand': 40, 'person:rlarm': 41, 'person:rhand': 42, 'person:rlleg': 43, 'person:ruleg': 44, 'person:lfoot': 45, 'person:rfoot': 46, 'motorbike': 47, 'motorbike:fwheel': 48, 'motorbike:bwheel': 49, 'car:frontside': 50, 'car:leftside': 51, 'car:wheel_1': 52, 'car:wheel_2': 53, 'bicycle': 54, 'bicycle:fwheel': 55, 'bicycle:bwheel': 56, 'bicycle:chainwheel': 57, 'bicycle:handlebar': 58, 'bicycle:saddle': 59, 'dog': 60, 'dog:head': 61, 'dog:rear': 62, 'dog:muzzle': 63, 'dog:nose': 64, 'dog:torso': 65, 'dog:neck': 66, 'dog:lfleg': 67, 'dog:lfpa': 68, 'dog:rfleg': 69, 'dog:rfpa': 70, 'dog:lbleg': 71, 'dog:lbpa': 72, 'dog:rbleg': 73, 'dog:rbpa': 74, 'dog:tail': 75, 'dog:lear': 76, 'dog:leye': 77, 'cat': 78, 'cat:head': 79, 'cat:lear': 80, 'cat:rear': 81, 'cat:leye': 82, 'cat:reye': 83, 'cat:nose': 84, 'cat:torso': 85, 'cat:neck': 86, 'cat:lfleg': 87, 'cat:lfpa': 88, 'cat:rfleg': 89, 'cat:rfpa': 90, 'cat:lbleg': 91, 'cat:lbpa': 92, 'cat:rbleg': 93, 'cat:rbpa': 94, 'cat:tail': 95, 'train': 96, 'train:head': 97, 'train:headlight_1': 98, 'train:hfrontside': 99, 'train:hleftside': 100, 'bird': 101, 'bird:head': 102, 'bird:reye': 103, 'bird:beak': 104, 'bird:torso': 105, 'bird:neck': 106, 'bird:lwing': 107, 'bird:rwing': 108, 'bird:lleg': 109, 'bird:lfoot': 110, 'bird:rleg': 111, 'bird:rfoot': 112, 'dog:reye': 113, 'train:coach_1': 114, 'train:cleftside_1': 115, 'train:headlight_2': 116, 'train:headlight_3': 117, 'horse': 118, 'horse:torso': 119, 'horse:lflleg': 120, 'horse:lfuleg': 121, 'horse:rflleg': 122, 'horse:rfuleg': 123, 'horse:head': 124, 'horse:lear': 125, 'horse:rear': 126, 'horse:leye': 127, 'horse:muzzle': 128, 'horse:neck': 129, 'horse:rblleg': 130, 'horse:rbuleg': 131, 'bottle': 132, 'bottle:body': 133, 'bottle:cap': 134, 'pottedplant': 135, 'pottedplant:pot': 136, 'pottedplant:plant': 137, 'car:door_1': 138, 'car:door_2': 139, 'train:crightside_1': 140, 'horse:reye': 141, 'horse:lfho': 142, 'horse:lblleg': 143, 'horse:lbuleg': 144, 'horse:tail': 145, 'horse:rfho': 146, 'horse:lbho': 147, 'car:leftmirror': 148, 'car:headlight_1': 149, 'car:headlight_2': 150, 'car:wheel_3': 151, 'car:wheel_4': 152, 'car:fliplate': 153, 'car:window_3': 154, 'bird:tail': 155, 'aeroplane': 156, 'aeroplane:body': 157, 'aeroplane:lwing': 158, 'aeroplane:stern': 159, 'aeroplane:rwing': 160, 'aeroplane:wheel_1': 161, 'aeroplane:wheel_2': 162, 'aeroplane:engine_1': 163, 'aeroplane:engine_2': 164, 'aeroplane:wheel_3': 165, 'aeroplane:tail': 166, 'table': 167, 'bird:leye': 168, 'train:hrightside': 169, 'cow:leye': 170, 'cow:muzzle': 171, 'cow:lblleg': 172, 'cow:rblleg': 173, 'cow:rbuleg': 174, 'boat': 175, 'motorbike:headlight_1': 176, 'sheep': 177, 'sheep:head': 178, 'sheep:torso': 179, 'sheep:neck': 180, 'sheep:tail': 181, 'sheep:muzzle': 182, 'sheep:lflleg': 183, 'sheep:lfuleg': 184, 'sheep:rflleg': 185, 'sheep:rfuleg': 186, 'sheep:lblleg': 187, 'sheep:rblleg': 188, 'sheep:rbuleg': 189, 'sheep:lear': 190, 'sheep:rear': 191, 'sheep:lbuleg': 192, 'bus': 193, 'bus:fliplate': 194, 'bus:frontside': 195, 'bus:rightside': 196, 'bus:leftmirror': 197, 'bus:rightmirror': 198, 'bus:headlight_1': 199, 'bus:headlight_2': 200, 'bus:headlight_3': 201, 'bus:headlight_4': 202, 'bus:wheel_1': 203, 'bus:wheel_2': 204, 'bus:window_1': 205, 'bus:window_2': 206, 'bus:window_3': 207, 'bus:window_4': 208, 'bus:window_5': 209, 'bus:window_6': 210, 'bus:window_7': 211, 'bus:window_8': 212, 'bus:window_9': 213, 'bus:window_10': 214, 'car:roofside': 215, 'sheep:reye': 216, 'bus:leftside': 217, 'bus:door_1': 218, 'bus:headlight_5': 219, 'bus:headlight_6': 220, 'sheep:leye': 221, 'car:window_4': 222, 'sofa': 223, 'bus:bliplate': 224, 'bus:backside': 225, 'bus:door_2': 226, 'bus:wheel_3': 227, 'train:coach_2': 228, 'train:coach_3': 229, 'train:coach_4': 230, 'train:coach_5': 231, 'train:coach_6': 232, 'train:coach_7': 233, 'train:coach_8': 234, 'train:cleftside_2': 235, 'train:cleftside_3': 236, 'train:cleftside_4': 237, 'train:cleftside_5': 238, 'train:cleftside_8': 239, 'train:crightside_6': 240, 'train:crightside_7': 241, 'train:crightside_2': 242, 'train:cbackside_1': 243, 'horse:rbho': 244, 'train:hroofside': 245, 'sheep:lhorn': 246, 'sheep:rhorn': 247, 'aeroplane:engine_3': 248, 'aeroplane:engine_4': 249, 'motorbike:headlight_2': 250, 'cow:rflleg': 251, 'cow:rhorn': 252, 'cow:lhorn': 253, 'aeroplane:wheel_4': 254, 'aeroplane:wheel_5': 255, 'aeroplane:wheel_6': 256, 'bus:window_11': 257, 'cow:reye': 258, 'bus:wheel_4': 259, 'train:croofside_1': 260, 'train:cfrontside_2': 261, 'aeroplane:wheel_7': 262, 'train:cfrontside_1': 263, 'train:headlight_4': 264, 'bus:roofside': 265, 'bus:headlight_7': 266, 'bus:headlight_8': 267, 'bus:window_12': 268, 'bus:window_13': 269, 'motorbike:handlebar': 270, 'motorbike:saddle': 271, 'bus:door_3': 272, 'car:window_5': 273, 'motorbike:headlight_3': 274, 'aeroplane:wheel_8': 275, 'car:headlight_3': 276, 'car:headlight_4': 277, 'car:headlight_5': 278, 'train:hbackside': 279, 'train:crightside_3': 280, 'train:croofside_2': 281, 'train:croofside_3': 282, 'bus:wheel_5': 283, 'car:window_6': 284, 'car:window_7': 285, 'train:cbackside_2': 286, 'car:headlight_6': 287, 'train:crightside_4': 288, 'train:crightside_5': 289, 'train:crightside_8': 290, 'train:coach_9': 291, 'train:cleftside_6': 292, 'train:cleftside_7': 293, 'train:cleftside_9': 294, 'train:cfrontside_3': 295, 'train:cfrontside_4': 296, 'train:cfrontside_5': 297, 'train:cfrontside_6': 298, 'train:cfrontside_7': 299, 'train:cfrontside_9': 300, 'train:headlight_5': 301, 'car:door_3': 302, 'bus:window_14': 303, 'train:croofside_4': 304, 'bus:window_15': 305, 'train:croofside_5': 306, 'car:wheel_5': 307, 'bicycle:headlight_1': 308, 'aeroplane:engine_5': 309, 'aeroplane:engine_6': 310, 'bus:window_16': 311, 'bus:window_17': 312, 'bus:window_18': 313, 'bus:window_19': 314, 'bus:door_4': 315, 'bus:window_20': 316}
"""

if __name__ == "__main__":
    main()
