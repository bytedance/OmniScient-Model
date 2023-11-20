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
    "--num_files_per_shard",
    type=int,
    default=1000,
)
args = arg_parser.parse_args()


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    all_images = glob.glob(args.image_dir + '/train/*.JPEG')
    import random
    random.seed(20230731)
    random.shuffle(all_images)

    with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
        for idx in range(len(all_images)):
            image_file = all_images[idx]
            part_ann_file = image_file.replace("/images/", "/annotations/").replace(".JPEG", ".png")
            whole_ann_file = part_ann_file.replace("/train/", "/train_whole/")
            sample_data = {}
            img = Image.open(image_file).convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            sample_data["image_base64"] = img_str.decode("utf-8")
            # convert to base64
            part_seg = Image.open(part_ann_file)
            buffered = BytesIO()
            part_seg.save(buffered, format="PNG")
            part_seg_str = base64.b64encode(buffered.getvalue())
            sample_data["part_seg_base64"] = part_seg_str.decode("utf-8")

            whole_seg = Image.open(whole_ann_file)
            buffered = BytesIO()
            whole_seg.save(buffered, format="PNG")
            whole_seg_str = base64.b64encode(buffered.getvalue())
            sample_data["whole_seg_base64"] = whole_seg_str.decode("utf-8")

            key_str = uuid.uuid4().hex
            sink.write({"__key__": key_str, "json": sample_data})

            if (idx + 1) % args.num_files_per_shard == 0:
                sink.next_stream()


if __name__ == "__main__":
    main()
