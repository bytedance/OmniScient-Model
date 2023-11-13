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

from .osm import OmniScientModel
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch


def create_model_and_transforms(base_model_name="Salesforce/instructblip-vicuna-7b"):
    base_model = InstructBlipForConditionalGeneration.from_pretrained(base_model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    model = OmniScientModel(base_model=base_model)
    processor = InstructBlipProcessor.from_pretrained(base_model_name)
    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    model.window_pos_embed.requires_grad_(True)
    model.mask_query.requires_grad_(True)
    model.context_query.requires_grad_(True)
    model.qformer_mode_query.requires_grad_(True)
    model.mode_query.requires_grad_(True)
    model.base_model.qformer.requires_grad_(True)
    model.base_model.language_projection.requires_grad_(True)
    return model, processor