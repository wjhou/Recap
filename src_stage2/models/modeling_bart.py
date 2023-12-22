from typing import Optional, Tuple, Dict, Any, Union, List
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    ViTConfig,
    BartForCausalLM,
    BartConfig,
    BartPretrainedModel,
)
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutput,
)
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartEncoder,
    BartDecoderLayer,
    BartDecoder,
    BartModel,
    _make_causal_mask,
    _expand_mask,
    BartAttention,
)
from transformers.utils import logging
from dataclasses import dataclass
import os
from transformers.file_utils import WEIGHTS_NAME
import random
from torch.nn import Embedding
from transformers.generation_utils import *

logger = logging.get_logger(__name__)


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    gate: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ViTBartOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    pre_visual_last_hidden_state: torch.FloatTensor = None
    progression_hidden_state: torch.FloatTensor = None
    progression_attention_mask: torch.FloatTensor = None
    observation_hidden_state: torch.FloatTensor = None
    observation_attention_mask: torch.FloatTensor = None
    node_hidden_state: torch.FloatTensor = None
    observation_det_logits: torch.FloatTensor = None
    observation_cls_logits: torch.FloatTensor = None
    progression_logits: torch.FloatTensor = None
    temporal_mask: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None


@dataclass
class CausalModelOutput(ModelOutput):
    encoder_loss: Optional[torch.FloatTensor] = None
    decoder_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attention_mask: Optional[Tuple[torch.FloatTensor]] = None
    encoder_visual_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_visual_attention_mask: Optional[Tuple[torch.FloatTensor]] = None
    node_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[Tuple[torch.FloatTensor]] = None


class BartEncoderCustom(BartEncoder):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # position_ids = ((attention_mask > 0).float().cumsum(dim=-1) - 1).long()
        position_ids = (attention_mask.cumsum(dim=-1) - 1).long()
        embed_pos = super(type(self.embed_positions), self.embed_positions).forward(
            self.embed_positions.offset + position_ids
        )
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # expand attention_mask
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (
                dropout_probability < self.layerdrop
            ):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class BartDecoderLayerCustom(BartDecoderLayer):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.encoder_visual_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_visual_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.progression_gate = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_visual_hidden_states: Optional[torch.Tensor] = None,
        encoder_visual_attention_mask: Optional[torch.Tensor] = None,
        temporal_mask=None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        index = 2

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # print("observation", encoder_hidden_states.size())
            residual = hidden_states
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[index : index + 2]
                if past_key_value is not None
                else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value
            index += 2

        cross_visual_attn_present_key_value = None
        cross_visual_attn_weights = None
        if encoder_visual_hidden_states is not None:
            # print("progression", encoder_visual_hidden_states.size())
            residual = hidden_states
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_visual_attn_past_key_value = (
                past_key_value[index : index + 2]
                if past_key_value is not None
                else None
            )
            (
                hidden_states,
                cross_visual_attn_weights,
                cross_visual_attn_present_key_value,
            ) = self.encoder_visual_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_visual_hidden_states,
                attention_mask=encoder_visual_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_visual_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = self.encoder_visual_attn_layer_norm(hidden_states)
            alpha = self.progression_gate(residual) * temporal_mask.unsqueeze(
                -1
            ).unsqueeze(-1)
            hidden_states = alpha * hidden_states + (1 - alpha) * residual

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_visual_attn_present_key_value
            index += 2

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (
                self_attn_weights,
                cross_attn_weights,
                cross_visual_attn_weights,
            )

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartDecoderCustom(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [BartDecoderLayerCustom(config) for _ in range(config.decoder_layers)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_visual_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_visual_attention_mask: Optional[torch.LongTensor] = None,
        temporal_mask=None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        if (
            encoder_visual_hidden_states is not None
            and encoder_visual_attention_mask is not None
        ):
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_visual_attention_mask = _expand_mask(
                encoder_visual_attention_mask,
                inputs_embeds.dtype,
                tgt_len=input_shape[-1],
            )

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_visual_hidden_states=encoder_visual_hidden_states,
                    encoder_visual_attention_mask=encoder_visual_attention_mask,
                    temporal_mask=temporal_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BartForCausalLMCustom(BartForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model.decoder = BartDecoderCustom(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_visual_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_visual_attention_mask: Optional[torch.FloatTensor] = None,
        temporal_mask=None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_visual_hidden_states=encoder_visual_hidden_states,
            encoder_visual_attention_mask=encoder_visual_attention_mask,
            temporal_mask=temporal_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            # logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class ViTEncoder(BartPretrainedModel):
    def __init__(self, config, decoder_config, embed_tokens):
        super().__init__(config)
        self.observation_bart = BartEncoderCustom(
            config=decoder_config, embed_tokens=embed_tokens
        )
        self.progression_bart = BartEncoderCustom(
            config=decoder_config, embed_tokens=embed_tokens
        )
        # 0 for current image
        # 1 for prior image
        # 2 for prior report
        # 3 for observation

        from models.rgcn import RGCN

        self.rgcn = RGCN(config)
        self.post_init()

        from src_stage1.models.modeling_vit import VisualEncoder

        self.vit = VisualEncoder(config=config)
        self.vit_config: ViTConfig = self.vit.visual_extractor.config
        if config.stage1_model_name_or_path is not None:
            print("***************************")
            print("***************************")
            print(
                "Loading Stage 1 Pretrained ViT Model", config.stage1_model_name_or_path
            )
            print("***************************")
            print("***************************")
            state_dict = torch.load(
                os.path.join(
                    config.stage1_model_name_or_path,
                    WEIGHTS_NAME,  # pytorch_model.bin
                ),
                map_location=self.device,
            )
            self.vit.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        progression_input_ids=None,
        progression_attention_mask=None,
        input_pixels=None,
        input_temporal_pixels=None,
        temporal_mask=None,
        observations=None,
        progressions=None,
        matrix=None,
        nodes=None,
        node_mask=None,
    ):
        loss = None
        node_hidden_state = None
        observation_det_logits = None
        observation_cls_logits = None
        progression_logits = None
        progression_hidden_state = None
        observation_hidden_state = None
        prior_attention_mask = None
        visual_attention_mask = None

        # spatiotemporal prediction
        if self.config.is_temporal == 0:
            temporal_mask = temporal_mask * 0
        if (
            input_pixels is not None
            and input_temporal_pixels is not None
            and temporal_mask is not None
        ):
            vit_outputs = self.vit(
                input_pixels=input_pixels,
                input_temporal_pixels=input_temporal_pixels,
                temporal_mask=temporal_mask,
                observations=observations,
                progressions=progressions,
                require_logits=False,
            )
            pre_visual_last_hidden_state = vit_outputs.last_hidden_state

            last_hidden_state, prior_last_hidden_state = pre_visual_last_hidden_state
            visual_attention_mask = torch.ones_like(last_hidden_state[:, :, 0])
            prior_attention_mask = torch.ones_like(
                prior_last_hidden_state[:, :, 0]
            ) * temporal_mask.unsqueeze(-1)
            if (
                progression_input_ids is not None
                and progression_attention_mask.sum() > 0
            ):
                progression_input_embeds = self.progression_bart.embed_tokens(
                    progression_input_ids
                )
                prior_last_hidden_state = torch.cat(
                    (
                        prior_last_hidden_state,
                        progression_input_embeds,
                    ),
                    dim=1,
                )
                prior_attention_mask = torch.cat(
                    (prior_attention_mask, progression_attention_mask), dim=-1
                )
            progression_hidden_state = self.progression_bart(
                inputs_embeds=prior_last_hidden_state,
                attention_mask=prior_attention_mask,
            ).last_hidden_state

            if input_ids is not None and attention_mask.sum() > 0:
                input_embeds = self.observation_bart.embed_tokens(input_ids)
                last_hidden_state = torch.cat((last_hidden_state, input_embeds), dim=1)
                visual_attention_mask = torch.cat(
                    (visual_attention_mask, attention_mask), dim=-1
                )
            observation_hidden_state = self.observation_bart(
                inputs_embeds=last_hidden_state,
                attention_mask=visual_attention_mask,
            ).last_hidden_state

        # precise attribute modeling
        node_hidden_state = self.rgcn(
            nodes=nodes,
            matrix=matrix,
        )

        return ViTBartOutput(
            loss=loss,
            pre_visual_last_hidden_state=pre_visual_last_hidden_state,
            progression_hidden_state=progression_hidden_state,
            progression_attention_mask=prior_attention_mask,
            observation_hidden_state=observation_hidden_state,
            observation_attention_mask=visual_attention_mask,
            node_hidden_state=node_hidden_state,
            observation_det_logits=observation_det_logits,
            observation_cls_logits=observation_cls_logits,
            progression_logits=progression_logits,
            temporal_mask=temporal_mask,
            pooler_output=vit_outputs.pooler_output,
        )


class ViTBartModel(BartPretrainedModel):
    def __init__(self, plm_config: BartConfig, init_config: BartConfig):
        super().__init__(plm_config)
        decoder = BartForCausalLMCustom(init_config)
        self.decoder = decoder.model.decoder
        self.lm_head = decoder.lm_head
        self.encoder = ViTEncoder(
            plm_config, init_config, embed_tokens=self.decoder.embed_tokens
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        progression_input_ids: torch.LongTensor = None,
        progression_attention_mask: torch.FloatTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        decoder_attention_mask: torch.FloatTensor = None,
        head_mask: torch.FloatTensor = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        input_pixels: torch.FloatTensor = None,
        input_temporal_pixels: torch.FloatTensor = None,
        temporal_mask: torch.FloatTensor = None,
        matrix: torch.FloatTensor = None,
        nodes: torch.LongTensor = None,
        node_mask: torch.FloatTensor = None,
        encoder_outputs: Optional[ModelOutput] = None,
        labels: Optional[torch.LongTensor] = None,
        observations: Optional[torch.FloatTensor] = None,
        progressions: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                progression_input_ids=progression_input_ids,
                progression_attention_mask=progression_attention_mask,
                input_pixels=input_pixels,
                input_temporal_pixels=input_temporal_pixels,
                temporal_mask=temporal_mask,
                observations=observations,
                progressions=progressions,
                matrix=matrix,
                nodes=nodes,
                node_mask=node_mask,
            )
        encoder_visual_hidden_states = None
        encoder_visual_attention_mask = None
        if self.config.is_temporal == 1:
            encoder_visual_hidden_states = encoder_outputs.progression_hidden_state
            encoder_visual_attention_mask = encoder_outputs.progression_attention_mask

        decoder_outputs = self.decoder(
            # self-attention
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            # observation-aware cross-attention
            encoder_hidden_states=encoder_outputs.observation_hidden_state,
            encoder_attention_mask=encoder_outputs.observation_attention_mask,
            # progression-aware cross-attention
            encoder_visual_hidden_states=encoder_visual_hidden_states,
            encoder_visual_attention_mask=encoder_visual_attention_mask,
            temporal_mask=encoder_outputs.temporal_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return CausalModelOutput(
            encoder_loss=encoder_outputs.loss,
            past_key_values=decoder_outputs.past_key_values,
            last_hidden_state=decoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.observation_hidden_state,
            encoder_attention_mask=encoder_outputs.observation_attention_mask,
            encoder_visual_hidden_states=encoder_outputs.progression_hidden_state,
            encoder_visual_attention_mask=encoder_outputs.progression_attention_mask,
            node_hidden_state=encoder_outputs.node_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
        )


class PrRModule(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.obs_weight = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_model, bias=False)
                for _ in range(config.num_relation - 1)
            ]
        )
        self.tok_weight = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_model, bias=False)
                for _ in range(config.num_relation - 1)
            ]
        )
        self.self_weight = nn.Linear(config.d_model, config.d_model, bias=False)
        self.act = torch.nn.Tanh()
        self.config = config

    def forward(
        self,
        last_hidden_state,
        node_hidden_state,
        cls_hidden_state,
        matrix,
        node_mask,
        nodes,
        gate_labels=None,
    ):
        node_matrix = matrix[:, :-1]
        value = node_hidden_state
        value = value.transpose(1, 2)
        obs_query = [weight_fn(last_hidden_state) for weight_fn in self.obs_weight]
        tok_query = [weight_fn(last_hidden_state) for weight_fn in self.tok_weight]
        obs_logits = torch.stack([q.bmm(value) for q in obs_query], dim=1)
        tok_logits = torch.stack([q.bmm(value) for q in tok_query], dim=1)
        prr_logits = tok_logits.unsqueeze(-1) + obs_logits.unsqueeze(-2)
        prr_logits = self.act(prr_logits) * (node_matrix > 0).float().unsqueeze(2)
        norm = (
            (node_matrix > 0)
            .float()
            .sum(dim=1)
            .sum(dim=-1, keepdim=True)
            .transpose(1, 2)
        )
        norm = torch.max(norm, torch.ones_like(norm))
        prr_logits = prr_logits.sum(dim=1).sum(dim=-1) / norm
        logits = (self.self_weight(last_hidden_state)).bmm(value)
        logits = self.act(logits)
        weight = logits + prr_logits * 2
        weight = weight.masked_fill(node_mask.unsqueeze(1) <= 0, -10000)
        node_proba = torch.softmax(weight, dim=-1)
        return node_proba, node_proba


class ViTBartForGeneration(BartPretrainedModel):
    def __init__(self, encoder_config: BartConfig, decoder_config: BartConfig):
        super().__init__(decoder_config)
        self.config = decoder_config
        self.main_input_name = "input_pixels"
        self.model_parallel = False
        self.prr_model = PrRModule(decoder_config)
        # copy gate
        self.controller = nn.Sequential(
            nn.Linear(decoder_config.d_model, 1, bias=False),
            nn.Sigmoid(),
        )
        self.apply(self._init_weights)
        # ViT Pretrained Model dose not need init weights
        self.model = ViTBartModel(encoder_config, decoder_config)
        self.lm_head = self.model.lm_head
        self.tie_weights()

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def get_output_embeddings(self):
        return self.model.decoder.embed_tokens

    def get_input_embeddings(self):
        return self.model.encoder.observation_bart.embed_tokens

    def set_input_embeddings(self, value):
        self.model.encoder.observation_bart.embed_tokens = value
        self.model.encoder.progression_bart.embed_tokens = value

    def tie_weights(self):
        return super().tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        progression_input_ids: torch.LongTensor = None,
        progression_attention_mask: torch.FloatTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        decoder_attention_mask: torch.FloatTensor = None,
        head_mask: torch.FloatTensor = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        input_pixels: torch.FloatTensor = None,
        input_temporal_pixels: torch.FloatTensor = None,
        temporal_mask: torch.FloatTensor = None,
        matrix: torch.FloatTensor = None,
        nodes: torch.LongTensor = None,
        node_mask: torch.FloatTensor = None,
        gather_index: torch.LongTensor = None,
        gate_labels: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        observations: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        progressions: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            progression_input_ids=progression_input_ids,
            progression_attention_mask=progression_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            input_pixels=input_pixels,
            input_temporal_pixels=input_temporal_pixels,
            temporal_mask=temporal_mask,
            encoder_outputs=encoder_outputs,
            matrix=matrix,
            nodes=nodes,
            node_mask=node_mask,
            labels=labels,
            observations=observations,
            progressions=progressions,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state
        lm_logits = self.lm_head(last_hidden_state)

        # Progression Reasoning (RrR)
        gate, proba = self.prr(
            lm_logits=lm_logits,
            outputs=outputs,
            gather_index=gather_index,
            node_mask=node_mask,
            matrix=matrix,
            gate_labels=gate_labels,
            nodes=nodes,
        )
        loss = None
        if labels is not None:
            loss = self.prr_loss(
                gate=gate,
                gate_labels=gate_labels,
                proba=proba,
                labels=labels,
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=proba,
            past_key_values=outputs.past_key_values,
        )

    def prr(
        self,
        lm_logits,
        outputs,
        gather_index,
        node_mask,
        matrix,
        gate_labels=None,
        nodes=None,
    ):
        node_proba, node_weight = self.prr_model(
            last_hidden_state=outputs.last_hidden_state,
            node_hidden_state=outputs.node_hidden_state,
            cls_hidden_state=outputs.pooler_output,
            matrix=matrix,
            node_mask=node_mask,
            nodes=nodes,
            gate_labels=gate_labels,
        )
        node_proba_vocab = node_proba.gather(
            -1, gather_index.unsqueeze(1).expand_as(lm_logits)
        )
        # 0 represents observation
        node_proba_vocab.masked_fill_(gather_index.unsqueeze(1) == 0, 0)

        gate_rep = outputs.last_hidden_state
        gate = self.controller(gate_rep)
        gate_mask = (node_mask.sum(dim=-1, keepdim=True) > 0).float().unsqueeze(1)
        gate = gate * gate_mask
        proba_vocab = torch.softmax(lm_logits, dim=-1)
        proba = (1.0 - gate) * proba_vocab + gate * node_proba_vocab
        proba = proba.clamp(min=1e-5, max=1.0 - 1e-5)
        return gate, proba

    def prr_loss(self, gate, gate_labels, proba, labels):
        loss_fct = nn.NLLLoss()
        loss = loss_fct(
            input=proba.log().view(-1, proba.size(-1)),
            target=labels.view(-1),
        )
        gate = gate.clamp(min=1e-5, max=1.0 - 1e-5)
        gate_mask = gate_labels != -100
        gate_labels = gate_labels.masked_fill(~gate_mask, 0)
        gate = gate.squeeze(-1)
        pointer_loss = (
            -(gate_labels * gate.log() + (1.0 - gate_labels) * (1 - gate).log())
            * gate_mask
        ).mean()
        if gate_mask.sum() > 0:
            loss = loss + pointer_loss * self.config.lambda_
        return loss

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,  # decoder_input_ids
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )
        if "temporal_mask" in model_kwargs:
            temporal_mask = model_kwargs["temporal_mask"]
            model_kwargs["temporal_mask"] = temporal_mask.index_select(
                0, expanded_return_idx
            )
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs[
                "decoder_attention_mask"
            ] = decoder_attention_mask.index_select(0, expanded_return_idx)
        if (
            "attention_mask" in model_kwargs
            and model_kwargs["attention_mask"] is not None
        ):
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )
        if "node_mask" in model_kwargs:
            node_mask = model_kwargs["node_mask"]
            model_kwargs["node_mask"] = node_mask.index_select(0, expanded_return_idx)

        if "gather_index" in model_kwargs:
            gather_index = model_kwargs["gather_index"]
            model_kwargs["gather_index"] = gather_index.index_select(
                0, expanded_return_idx
            )

        if "matrix" in model_kwargs:
            matrix = model_kwargs["matrix"]
            model_kwargs["matrix"] = matrix.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            if (
                "last_hidden_state" in encoder_outputs
                and encoder_outputs["last_hidden_state"] is not None
            ):
                encoder_outputs["last_hidden_state"] = encoder_outputs[
                    "last_hidden_state"
                ].index_select(0, expanded_return_idx)
            if (
                "visual_last_hidden_state" in encoder_outputs
                and encoder_outputs["visual_last_hidden_state"] is not None
            ):
                encoder_outputs["visual_last_hidden_state"] = encoder_outputs[
                    "visual_last_hidden_state"
                ].index_select(0, expanded_return_idx)
            if (
                "visual_attention_mask" in encoder_outputs
                and encoder_outputs["visual_attention_mask"] is not None
            ):
                encoder_outputs["visual_attention_mask"] = encoder_outputs[
                    "visual_attention_mask"
                ].index_select(0, expanded_return_idx)
            if (
                "node_hidden_state" in encoder_outputs
                and encoder_outputs["node_hidden_state"] is not None
            ):
                encoder_outputs["node_hidden_state"] = encoder_outputs[
                    "node_hidden_state"
                ].index_select(0, expanded_return_idx)
            if (
                "pooler_output" in encoder_outputs
                and encoder_outputs["pooler_output"] is not None
            ):
                encoder_outputs["pooler_output"] = encoder_outputs[
                    "pooler_output"
                ].index_select(0, expanded_return_idx)
            if (
                "progression_hidden_state" in encoder_outputs
                and encoder_outputs["progression_hidden_state"] is not None
            ):
                encoder_outputs["progression_hidden_state"] = encoder_outputs[
                    "progression_hidden_state"
                ].index_select(0, expanded_return_idx)
                encoder_outputs["progression_attention_mask"] = encoder_outputs[
                    "progression_attention_mask"
                ].index_select(0, expanded_return_idx)
            if (
                "observation_hidden_state" in encoder_outputs
                and encoder_outputs["observation_hidden_state"] is not None
            ):
                encoder_outputs["observation_hidden_state"] = encoder_outputs[
                    "observation_hidden_state"
                ].index_select(0, expanded_return_idx)
                encoder_outputs["observation_attention_mask"] = encoder_outputs[
                    "observation_attention_mask"
                ].index_select(0, expanded_return_idx)
                encoder_outputs["temporal_mask"] = encoder_outputs[
                    "temporal_mask"
                ].index_select(0, expanded_return_idx)
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past

    def prepare_inputs_for_generation(
        self,
        # attention_mask,
        decoder_input_ids,
        decoder_attention_mask=None,
        past=None,  # substitute to `past` in transformers==4.15.0
        temporal_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        node_mask=None,
        nodes=None,
        gather_index=None,
        matrix=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "attention_mask": kwargs.get("attention_mask", None),
            "decoder_input_ids": decoder_input_ids,
            "head_mask": head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "temporal_mask": temporal_mask,
            # "decoder_attention_mask": decoder_attention_mask,
            # change this to avoid caching (presumably for debugging)
            "use_cache": use_cache,
            "node_mask": node_mask,
            "nodes": nodes,
            "gather_index": gather_index,
            "matrix": matrix,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning,
            )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        output_scores = (
            output_scores if output_scores is not None else self.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size))
            if (return_dict_in_generate and output_scores)
            else None
        )
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # NOTICE major revision of beam_search
            next_token_scores = next_token_logits.log()

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[
                :, None
            ].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(
                    model_kwargs["past"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple(
                    (
                        beam_indices[beam_idx[i]] + (beam_idx[i],)
                        for i in range(len(beam_indices))
                    )
                )

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
