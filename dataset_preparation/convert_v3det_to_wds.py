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
import logging
from collections import defaultdict


class V3Det:
    def __init__(self, annotation_path):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading annotations.")

        self.dataset = self._load_json(annotation_path)

        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))
        self._create_index()

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _create_index(self):
        self.logger.info("Creating index.")

        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)

        self.anns = {}
        self.cats = {}
        self.imgs = {}

        for ann in self.dataset["annotations"]:
            self.img_ann_map[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann

        for img in self.dataset["images"]:
            self.imgs[img["id"]] = img

        for cat in self.dataset["categories"]:
            self.cats[cat["id"]] = cat

        for ann in self.dataset["annotations"]:
            self.cat_img_map[ann["category_id"]].append(ann["image_id"])

        self.logger.info("Index created.")

    def get_ann_ids(self, img_ids=None, cat_ids=None, area_rng=None):
        """Get ann ids that satisfy given filter conditions.
        Args:
            img_ids (int array): get anns for given imgs
            cat_ids (int array): get anns for given cats
            area_rng (float array): get anns for a given area range. e.g [0, inf]
        Returns:
            ids (int array): integer array of ann ids
        """
        anns = []
        if img_ids is not None:
            for img_id in img_ids:
                anns.extend(self.img_ann_map[img_id])
        else:
            anns = self.dataset["annotations"]

        # return early if no more filtering required
        if cat_ids is None and area_rng is None:
            return [_ann["id"] for _ann in anns]

        cat_ids = set(cat_ids)

        if area_rng is None:
            area_rng = [0, float("inf")]

        ann_ids = [
            _ann["id"]
            for _ann in anns
            if _ann["category_id"] in cat_ids
            and _ann["area"] > area_rng[0]
            and _ann["area"] < area_rng[1]
        ]
        return ann_ids

    def get_cat_ids(self):
        """Get all category ids.
        Returns:
            ids (int array): integer array of category ids
        """
        return list(self.cats.keys())

    def get_img_ids(self):
        """Get all img ids.
        Returns:
            ids (int array): integer array of image ids
        """
        return list(self.imgs.keys())

    def _load_helper(self, _dict, ids):
        if ids is None:
            return list(_dict.values())
        else:
            return [_dict[id] for id in ids]

    def load_anns(self, ids=None):
        """Load anns with the specified ids. If ids=None load all anns.
        Args:
            ids (int array): integer array of annotation ids
        Returns:
            anns (dict array) : loaded annotation objects
        """
        return self._load_helper(self.anns, ids)

    def load_cats(self, ids):
        """Load categories with the specified ids. If ids=None load all
        categories.
        Args:
            ids (int array): integer array of category ids
        Returns:
            cats (dict array) : loaded category dicts
        """
        return self._load_helper(self.cats, ids)

    def load_imgs(self, ids):
        """Load categories with the specified ids. If ids=None load all images.
        Args:
            ids (int array): integer array of image ids
        Returns:
            imgs (dict array) : loaded image dicts
        """
        return self._load_helper(self.imgs, ids)

    # def ann_to_rle(self, ann):
    #     """Convert annotation which can be polygons, uncompressed RLE to RLE.
    #     Args:
    #         ann (dict) : annotation object
    #     Returns:
    #         ann (rle)
    #     """
    #     img_data = self.imgs[ann["image_id"]]
    #     h, w = img_data["height"], img_data["width"]
    #     segm = ann["segmentation"]
    #     if isinstance(segm, list):
    #         # polygon -- a single object might consist of multiple parts
    #         # we merge all parts into one mask rle code
    #         rles = mask_utils.frPyObjects(segm, h, w)
    #         rle = mask_utils.merge(rles)
    #     elif isinstance(segm["counts"], list):
    #         # uncompressed RLE
    #         rle = mask_utils.frPyObjects(segm, h, w)
    #     else:
    #         # rle
    #         rle = ann["segmentation"]
    #     return rle

    # def ann_to_mask(self, ann):
    #     """Convert annotation which can be polygons, uncompressed RLE, or RLE
    #     to binary mask.
    #     Args:
    #         ann (dict) : annotation object
    #     Returns:
    #         binary mask (numpy 2D array)
    #     """
    #     rle = self.ann_to_rle(ann)
    #     return mask_utils.decode(rle)


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
    open_api = V3Det(args.json_file)
    img_ids = sorted(open_api.imgs.keys())
    imgs = open_api.load_imgs(img_ids)
    anns = [open_api.img_ann_map[img_id] for img_id in img_ids]
    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        args.json_file
    )
    imgs_anns = list(zip(imgs, anns))
    print("Loaded {} images in the V3Det format from {}".format(len(imgs_anns), args.json_file))
    import random
    random.seed(20230731)
    random.shuffle(imgs_anns)

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        # example: {'file_name': 'images/n11735977/33_287_14396504429_d787b4291b_c.jpg',
        #  'height': 799,
        #  'width': 678,
        #  'id': 1}
        record["file_name"] = os.path.join(args.image_dir, img_dict['file_name'])
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
            objs.append(obj)
        record["annotations"] = objs
        if len(objs) > 0:
            dataset_dicts.append(record)


    with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
        for idx in range(len(dataset_dicts)):
            cur_dataset_dict = dataset_dicts[idx]
            image_file = os.path.join(args.image_dir, cur_dataset_dict["file_name"])
            if not os.path.exists(image_file):
                print(f"{image_file} is missing!")
                continue
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
