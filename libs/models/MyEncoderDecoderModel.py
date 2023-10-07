"""
自定义的encoder  decoder model
用于 多模态生成式模型
"""

import gc
import inspect
import os
import tempfile
import warnings
from typing import Optional, Tuple, Union

from transformers import EncoderDecoderModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

_CONFIG_FOR_DOC = "EncoderDecoderConfig"

DEPRECATION_WARNING = (
    "Version v4.12.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.12.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)

ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
            
        bbox (`torch.LongTensor` of shape `({0}, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.
            
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Batch of document images. Each image is divided into patches of shape `(num_channels, config.patch_size,
            config.patch_size)` and the total number of patches (=`patch_sequence_length`) equals to `((height /
            config.patch_size) * (width / config.patch_size))`.
            
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the
            right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):
            This tuple must consist of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) is a tensor
            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0,
            ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.Seq2SeqLMOutput`] instead of a plain tuple.
        kwargs (*optional*): Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:

            - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
            - With a *decoder_* prefix which will be input as `**decoder_kwargs` for the decoder forward function.
"""

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class MyEncoderDecoderModel(EncoderDecoderModel):
    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            bbox: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,

            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
            past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,

            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import EncoderDecoderModel, BertTokenizer
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "bert-base-uncased", "bert-base-uncased"
        ... )  # initialize Bert2Bert from pre-trained checkpoints

        >>> # training
        >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
        >>> model.config.pad_token_id = tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
        >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss, logits = outputs.loss, outputs.logits

        >>> # save and load from pretrained
        >>> model.save_pretrained("bert2bert")
        >>> model = EncoderDecoderModel.from_pretrained("bert2bert")

        >>> # generation
        >>> generated = model.generate(input_ids)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        # todo:
        #  v3 不好直接改
        #  不改动V3那就要改动这块，做好attn的mask的变更, 或者输出的截断;
        #  目前还是倾向于带上图片部分的hidden; 先调通，性能问题后续再研究
        #  HARD  HARDCODE !
        if pixel_values is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
            visual_embeddings_length = 197
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings_length), dtype=torch.long, device=device
            )
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            else:
                attention_mask = visual_attention_mask

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

