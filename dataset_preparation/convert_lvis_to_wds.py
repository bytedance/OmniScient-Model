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
from lvis import LVIS

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
    lvis_api = LVIS(args.json_file)
    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        args.json_file
    )
    imgs_anns = list(zip(imgs, anns))
    print("Loaded {} images in the LVIS format from {}".format(len(imgs_anns), args.json_file))

    import random
    random.seed(20230731)
    random.shuffle(imgs_anns)

    def get_file_name(img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(img_root + split_folder, file_name)
    
    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = get_file_name(args.image_dir, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            # XYWH_ABS: (x0, y0, w, h) in absolute floating points coordinates.
            # obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            obj = {"bbox": anno["bbox"]}
            obj["category_id"] = anno["category_id"]
            segm = anno["segmentation"]  # list[list[float]]
            # filter out invalid polygons (< 3 points)
            valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            assert len(segm) == len(
                valid_segm
            ), "Annotation contains an invalid polygon with < 3 points"
            assert len(segm) > 0
            obj["segmentation"] = segm
            objs.append(obj)
        record["annotations"] = objs
        if len(objs) > 0:
            dataset_dicts.append(record)


    with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
        for idx in range(len(dataset_dicts)):
            cur_dataset_dict = dataset_dicts[idx]
            image_file = os.path.join(args.image_dir, cur_dataset_dict["file_name"])
            sample_data = {}
            sample_data["segments_info"] = cur_dataset_dict["annotations"]
            sample_data["height"] = cur_dataset_dict["height"]
            sample_data["width"] = cur_dataset_dict["width"]
            img = Image.open(image_file).convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            # convert to base64
            sample_data["image_base64"] = img_str.decode("utf-8")
            key_str = uuid.uuid4().hex
            sink.write({"__key__": key_str, "json": sample_data})

            if (idx + 1) % args.num_files_per_shard == 0:
                sink.next_stream()


if __name__ == "__main__":
    main()
