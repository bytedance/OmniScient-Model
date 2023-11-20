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


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    # list of dict containing: "segments_info", "file_name" (ending in .png)
    # "segments_info" is a list of dict containing: id, category_id, bbox, area
    json_file = json.load(open(args.json_file, 'r'))["annotations"]
    import random
    random.seed(20230731)
    random.shuffle(json_file)
    with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
        for idx in range(len(json_file)):
            cur_json_file = json_file[idx]
            image_file = os.path.join(args.image_dir, cur_json_file["file_name"].split('_')[0], cur_json_file["file_name"].replace("_gtFine_panoptic.png", "_leftImg8bit.png"))
            pan_file = os.path.join(args.ann_dir, cur_json_file["file_name"])

            sample_data = {}
            segments_info = cur_json_file["segments_info"]
            if len(segments_info) < 1:
                continue
            sample_data["segments_info"] = segments_info

            img = Image.open(image_file).convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            # convert to base64
            sample_data["image_base64"] = img_str.decode("utf-8")

            pan = Image.open(pan_file).convert("RGB")
            buffered = BytesIO()
            pan.save(buffered, format="PNG") # lossless
            pan_str = base64.b64encode(buffered.getvalue())
            # convert to base64
            sample_data["pan_base64"] = pan_str.decode("utf-8")

            key_str = uuid.uuid4().hex
            sink.write({"__key__": key_str, "json": sample_data})

            if (idx + 1) % args.num_files_per_shard == 0:
                sink.next_stream()


if __name__ == "__main__":
    main()
