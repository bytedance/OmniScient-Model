# OmniScient-Model
 
This repo contains the code for our paper [**Towards Open-Ended Visual Recognition with Large Language Model**](https://arxiv.org/abs/2311.08400)

<div align="center">
  <img src="imgs/task.png" width="80%" height="80%"/>
</div><br/>

We propose OmniScient Model (OSM) towards open-ended visual recognition, allowing the identification of diverse real-world entities without the constraints of a user-defined vocabulary. Unlike closed-vocabulary and open-vocabulary recognition frameworks, OSM operates seamlessly without the need for predefined vocabularies.


<div align="center">
  <img src="imgs/teaser.png" width="70%" height="40%"/>
</div><br/>

### Features
* A simple strategy to adapt multi-modal LLM for high-resolution image at 1120x1120, leading to more precise recognition ability.

* A brand-new task named open-ended visual recognition to predict beyond the limitation of a given vocabulary.

* A strong model that can recognize novel concepts in the real-world, e.g., it can recognize semantic parts even when only trained on object-level data.

## Installation

```bash
pip install torch==2.0.1 torchvision==0.15.2
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -r requirments.txt
```

## Getting Started

We provide examples applying OSM on top of an off-the-shelf segmenter (e.g., SAM), illustrating playing with OSM in a segment and recognize anything mode in [demo_with_sam.py](./demo_with_sam.py), or in an interactive model in [interactive_demo.ipynb](./interactive_demo.ipynb).

Training and evaluation scripts will be released in the near future.


## Data Preparation

Please refer to [Preparing Datasets for OSM](dataset_preparation/README.md).

## Model Zoo

<table>
<thead>
  <tr>
    <th align="center" style="text-align:center">Checkpoint</th>
    <th align="center" style="text-align:center">Training Datasets</th>
  </tr>
</thead>
<tbody>
    <tr>
        <th align="center" style="text-align:center"><a href="https://drive.google.com/file/d/1ObaklM3NohoPIm_IaTde0RbCefZC8RtL/view?usp=drive_link"> OSM </th>
        <th align="center" style="text-align:center">COCO Panoptic, ADE Panoptic, Cityscapes Panoptic, LVIS Instance, A-847 Semantic, PC-459 Semantic</th>
    </tr>
    <tr>
        <th align="center" style="text-align:center"><a href="https://drive.google.com/file/d/1mGlvlfPR2MTTUuLpiklUrmSziBFai4-Z/view?usp=drive_link"> OSM w/ part and box</th>
        <th align="center" style="text-align:center">COCO Panoptic, ADE Panoptic, Cityscapes Panoptic, LVIS Instance, A-847 Semantic, PC-459 Semantic, Part-ImageNet Semantic, Pascal-Part Semantic, V3Det Detection</th>
    </tr>
</tbody>
</table>

## Visual Results

<div align="center">
  <img src="imgs/vis.png" width="100%" height="100%"/>
</div><br/>
<div align="center">
  <img src="imgs/vis2.png" width="100%" height="100%"/>
</div><br/>
<div align="center">
  <img src="imgs/vis3.png" width="100%" height="100%"/>
</div><br/>

## <a name="Citing OSM"></a>Citing OSM

If you use OSM in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{yu2023towards,
  title={Towards Open-Ended Visual Recognition with Large Language Model},
  author={Qihang Yu and Xiaohui Shen and Liang-Chieh Chen},
  booktitle={arxiv: 2311.08400},
  year={2023}
}
```

## Acknowledgement

[Segment Anything](https://github.com/facebookresearch/segment-anything)

[OpenFlamingo](https://github.com/mlfoundations/open_flamingo)
