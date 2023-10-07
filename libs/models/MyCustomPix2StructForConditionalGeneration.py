"""
简单 变更forward 方法
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructVisionModel, Pix2StructTextModel


class MyCustomPix2StructForConditionalGeneration(Pix2StructForConditionalGeneration):
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Pix2StructVisionModel) or isinstance(module, Pix2StructTextModel):
            module.gradient_checkpointing = value

    def forward(
            self,
            flattened_patches: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            decoder_inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # 占个位置，融入到 transformers 架构不会有异常
            input_ids=None,
            prompt=None,

    ):
        return super().forward(
            flattened_patches,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            labels,
            decoder_inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict
        )
