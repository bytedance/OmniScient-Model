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
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class OmniScientModel(nn.Module):
    def __init__(self, base_model, input_size=1120, sliding_window_size=224, sliding_window_stride=224,
                 backbone_output_stride=14, backbone_output_channel=1408, maskqformer_channel=768, llm_channel=4096):
        super().__init__()
        # we use InstructBLIP as our base model
        self.base_model = base_model
        self.cached_image_embeds = None
        self.input_size = input_size
        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride
        self.backbone_output_stride = backbone_output_stride
        self.backbone_output_channel = backbone_output_channel
        self.maskqformer_channel = maskqformer_channel
        self.llm_channel = llm_channel

        self.h_grids = max(input_size - sliding_window_size + sliding_window_stride - 1, 0) // sliding_window_stride + 1
        self.w_grids = max(input_size - sliding_window_size + sliding_window_stride - 1, 0) // sliding_window_stride + 1
        self.num_cls_tokens = self.h_grids * self.w_grids
        # To compensate the missing positional info in sliding window
        self.window_pos_embed = nn.Parameter(torch.randn(1, (input_size // backbone_output_stride) ** 2 + self.num_cls_tokens, backbone_output_channel))

        self.mask_query = nn.Parameter(torch.randn(1, 32, maskqformer_channel))
        trunc_normal_(self.mask_query, std=1.0)
        self.copy_weight_for_mask_query_tokens()
        self.context_query = self.base_model.query_tokens

        self.all_datasets = ["coco", "lvis", "v3det", "a847", "pc459", "partimagenet",
                              "pascal_part", "ade20k", "cityscapes", "any"]
        self.qformer_mode_query = nn.ParameterDict()
        self.mode_query = nn.ParameterDict()
        for k in self.all_datasets:
            self.qformer_mode_query[k] = nn.Parameter(torch.randn(1, 1, maskqformer_channel))
            self.mode_query[k] = nn.Parameter(torch.randn(1, 1, llm_channel))
            trunc_normal_(self.qformer_mode_query[k], std=.02)
            trunc_normal_(self.mode_query[k], std=.02)

    @torch.no_grad()
    def copy_weight_for_mask_query_tokens(self):
        self.mask_query.copy_(self.base_model.query_tokens.detach())

    @torch.no_grad()
    def sliding_window_vit_forward(self,
                            pixel_values):
        batch_size = pixel_values.shape[0]
        output_features = torch.zeros(
                    (batch_size, self.input_size // self.backbone_output_stride, self.input_size // self.backbone_output_stride, self.backbone_output_channel), dtype=pixel_values.dtype, device=pixel_values.device
                )
        counters = torch.zeros(
                    (batch_size, self.input_size // self.backbone_output_stride, self.input_size // self.backbone_output_stride, 1), dtype=pixel_values.dtype, device=pixel_values.device
                )
        
        all_cls_tokens = []
        for h_idx in range(self.h_grids):
            for w_idx in range(self.w_grids):
                y1 = h_idx * self.sliding_window_stride
                x1 = w_idx * self.sliding_window_stride
                y2 = min(y1 + self.sliding_window_size, self.input_size)
                x2 = min(x1 + self.sliding_window_size, self.input_size)
                y1 = max(y2 - self.sliding_window_size, 0)
                x1 = max(x2 - self.sliding_window_size, 0)
                cur_pixel_values = pixel_values[..., y1:y2, x1:x2]

                target_dtype = self.base_model.vision_model.embeddings.patch_embedding.weight.dtype
                patch_embeds = self.base_model.vision_model.embeddings.patch_embedding(cur_pixel_values)  # shape = [*, width, grid, grid]
                patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

                class_embeds = self.base_model.vision_model.embeddings.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
                embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
                position_embedding = self.base_model.vision_model.embeddings.position_embedding.to(target_dtype)
                hidden_states = embeddings + position_embedding
                last_hidden_state = self.base_model.vision_model.encoder(
                                    inputs_embeds=hidden_states)[0]
                last_hidden_state = self.base_model.vision_model.post_layernorm(last_hidden_state)
                last_hidden_state, cls_tokens = last_hidden_state[:, 1:], last_hidden_state[:, 0:1]
                output_features[:, y1//self.backbone_output_stride:y2//self.backbone_output_stride, x1//self.backbone_output_stride:x2//self.backbone_output_stride] += last_hidden_state.view(batch_size, self.sliding_window_size//self.backbone_output_stride, self.sliding_window_size//self.backbone_output_stride, -1)
                counters[:, y1//self.backbone_output_stride:y2//self.backbone_output_stride, x1//self.backbone_output_stride:x2//self.backbone_output_stride] += 1
                all_cls_tokens.append(cls_tokens)

        output_features /= counters
        encoded_pixel_features = output_features.view(batch_size, -1, self.backbone_output_channel)
        all_cls_tokens = torch.cat(all_cls_tokens, dim=1)
        encoded_pixel_features = torch.cat([all_cls_tokens, encoded_pixel_features], dim=1) # pre-pend all cls tokens

        return encoded_pixel_features

    @torch.no_grad()
    def generate_attention_mask(self, image_embeds, segmentation_mask, context_mask):
        batch_size = image_embeds.shape[0]
        attn_masks = F.interpolate(segmentation_mask.to(image_embeds.dtype),
                                               size=(self.input_size//self.backbone_output_stride, self.input_size//self.backbone_output_stride), mode='nearest').reshape(batch_size, -1).to(torch.long)
        context_attn_masks = F.interpolate(context_mask.to(image_embeds.dtype),
                                               size=(self.input_size//self.backbone_output_stride, self.input_size//self.backbone_output_stride), mode='nearest').reshape(batch_size, -1).to(torch.long)
        # pre-pend ones for cls tokens
        attn_masks = torch.cat([torch.ones_like(attn_masks[:, :self.num_cls_tokens]), attn_masks], dim=-1)
        context_attn_masks = torch.cat([torch.ones_like(context_attn_masks[:, :self.num_cls_tokens]), context_attn_masks], dim=-1)
        return attn_masks, context_attn_masks
    
    def maskqformer_encoding(self, image_embeds, image_attention_mask, image_context_attention_mask, dataset_type, qformer_input_ids, qformer_attention_mask):
        mask_query = self.mask_query.expand(image_embeds.shape[0], -1, -1)
        context_query = self.context_query.expand(image_embeds.shape[0], -1, -1)
        all_query = torch.cat([mask_query, context_query], dim=1)

        # append mode query
        qformer_mode_query = 0
        assert dataset_type in self.all_datasets
        if self.training:
            # This for loop avoids unused parameters in DDP training
            for dataset_type_key in self.all_datasets:
                if dataset_type_key == dataset_type:
                    qformer_mode_query += self.qformer_mode_query[dataset_type_key] * 1
                else:
                    qformer_mode_query += self.qformer_mode_query[dataset_type_key] * 0
        else:
            qformer_mode_query = self.qformer_mode_query[dataset_type]

        qformer_mode_query = qformer_mode_query.expand(all_query.shape[0], -1, -1)
        all_query = torch.cat([all_query, qformer_mode_query], dim=1)

        # cross-attention mask
        image_attention_mask = image_attention_mask[:, None, :].expand(-1, mask_query.shape[1], -1)
        image_context_attention_mask = image_context_attention_mask[:, None, :].expand(-1, all_query.shape[1] - mask_query.shape[1], -1)
        image_attention_mask = torch.cat([image_attention_mask, image_context_attention_mask], dim=1)

        query_attention_mask = torch.ones(all_query.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)

        query_outputs = self.base_model.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=all_query,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask
        )[0][:, :mask_query.size(1)] # we only keep mask queries as output
        
        return query_outputs

    def llm_decoding(self, qformer_output, dataset_type, input_ids, attention_mask, is_generation, **generate_kwargs):
        language_model_inputs = self.base_model.language_projection(qformer_output)

        # append dataset-specific query
        mode_query = 0
        assert dataset_type in self.all_datasets
        if self.training:
            # This for loop avoids unused parameters in DDP training
            for dataset_type_key in self.all_datasets:
                if dataset_type_key == dataset_type:
                    mode_query += self.mode_query[dataset_type_key] * 1
                else:
                    mode_query += self.mode_query[dataset_type_key] * 0
        else:
            mode_query = self.mode_query[dataset_type]
        mode_query = mode_query.expand(language_model_inputs.shape[0], -1, -1) # B x 1 x C
        language_model_inputs = torch.cat([language_model_inputs, mode_query], dim=1)

        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        assert self.base_model.config.use_decoder_only_language_model

        if is_generation:
            outputs = self.base_model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        else:
            outputs = self.base_model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask)

        return outputs

    def forward(
        self,
        pixel_values,
        qformer_input_ids = None,
        qformer_attention_mask = None,
        input_ids = None,
        attention_mask = None,
        labels = None,
        segmentation_mask = None,
        input_context_mask = None,
        dataset_type="any",
    ):

        image_embeds = self.sliding_window_vit_forward(pixel_values)
        image_embeds += self.window_pos_embed
        image_attention_mask, image_context_attention_mask = self.generate_attention_mask(image_embeds, segmentation_mask, input_context_mask)
        qformer_output = self.maskqformer_encoding(image_embeds, image_attention_mask, image_context_attention_mask, dataset_type, qformer_input_ids, qformer_attention_mask)
        llm_outputs = self.llm_decoding(qformer_output, dataset_type, input_ids, attention_mask, is_generation=False)
        logits = llm_outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits.view(-1, self.base_model.config.text_config.vocab_size), shift_labels.view(-1))
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        qformer_input_ids = None,
        qformer_attention_mask = None,
        input_ids = None,
        attention_mask = None,
        cache_image_embeds = False,
        segmentation_mask = None,
        input_context_mask = None,
        dataset_type="any",
        **generate_kwargs,
    ) -> torch.LongTensor:
        if hasattr(self.base_model, "hf_device_map"):
            # preprocess for `accelerate`
            self.base_model._preprocess_accelerate()

        if (not cache_image_embeds) and self.cached_image_embeds is not None:
            image_embeds = self.cached_image_embeds
        else:
            image_embeds = self.sliding_window_vit_forward(pixel_values)
            image_embeds += self.window_pos_embed
            if cache_image_embeds:
                self.cached_image_embeds = image_embeds

        image_attention_mask, image_context_attention_mask = self.generate_attention_mask(image_embeds, segmentation_mask, input_context_mask)
        qformer_output = self.maskqformer_encoding(image_embeds, image_attention_mask, image_context_attention_mask, dataset_type, qformer_input_ids, qformer_attention_mask)
        llm_outputs = self.llm_decoding(qformer_output, dataset_type, input_ids, attention_mask, is_generation=True, **generate_kwargs)

        if self.base_model.config.text_config.architectures[0] == "LLaMAForCausalLM":
            if generate_kwargs["return_dict_in_generate"]:
                llm_outputs["sequences"][llm_outputs["sequences"] == 0] = 2
            else:
                llm_outputs[llm_outputs == 0] = 2

        return llm_outputs