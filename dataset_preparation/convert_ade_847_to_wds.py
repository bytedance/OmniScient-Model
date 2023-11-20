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
    "--ann_dir",
    type=str
)
arg_parser.add_argument(
    "--image_dir",
    type=str,
    help="Pass in the directory where the images have been downloaded to.",
)
arg_parser.add_argument(
    "--json_file",
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

def main():
    os.makedirs(args.output_dir, exist_ok=True)

    file_list = glob.glob(args.image_dir + '/*.jpg')
    import random
    random.seed(20230731)
    random.shuffle(file_list)
    with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
        for idx in range(len(file_list)):
            image_file = file_list[idx]
            seg_file = image_file.replace(args.image_dir, args.ann_dir).replace('.jpg', '.tif')

            sample_data = {}
            img = Image.open(image_file).convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            # convert to base64
            sample_data["image_base64"] = img_str.decode("utf-8")

            seg = Image.open(seg_file)
            buffered = BytesIO()
            seg.save(buffered, format="TIFF")
            seg_str = base64.b64encode(buffered.getvalue())
            # convert to base64
            sample_data["seg_base64"] = seg_str.decode("utf-8")

            key_str = uuid.uuid4().hex
            sink.write({"__key__": key_str, "json": sample_data})

            if (idx + 1) % args.num_files_per_shard == 0:
                sink.next_stream()


if __name__ == "__main__":
    main()
