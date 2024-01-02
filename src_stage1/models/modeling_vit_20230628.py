from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torchvision.models as models
from transformers import PreTrainedModel, ViTModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from transformers import ViTConfig


@dataclass
class VisualOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    observation_loss: Optional[torch.FloatTensor] = None
    progression_loss: Optional[torch.FloatTensor] = None
    observation_det_logits: torch.FloatTensor = None
    observation_cls_logits: torch.FloatTensor = None
    progression_logits: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None


class VisualEncoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual_config = ViTConfig.from_pretrained(
            config.pretrained_visual_extractor
        )
        self.observation_det = nn.Linear(
            self.config.hidden_size, config.num_observation - 1
        )
        self.observation_cls = nn.Linear(
            self.config.hidden_size, config.num_observation
        )
        self.progression_cls = nn.Linear(
            self.config.hidden_size * 2, config.num_progression
        )
        self.post_init()
        self.visual_extractor = ViTModel.from_pretrained(
            config.pretrained_visual_extractor
        )

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def encode_image(self, input_pixels, require_logits=True):
        observation_det_logits = None
        observation_cls_logits = None
        visual_outputs = self.visual_extractor(input_pixels)
        pooler_output = visual_outputs.pooler_output
        last_hidden_state = visual_outputs.last_hidden_state
        if require_logits:
            observation_det_logits = self.observation_det(pooler_output)
            observation_cls_logits = self.observation_cls(pooler_output)
        return (
            pooler_output,
            last_hidden_state,
            observation_det_logits,
            observation_cls_logits,
        )

    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        input_temporal_pixels: torch.FloatTensor = None,
        temporal_mask: torch.FloatTensor = None,
        observations: Optional[torch.FloatTensor] = None,
        progressions: Optional[torch.FloatTensor] = None,
        entity_labels: Optional[torch.FloatTensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        require_logits=True,
    ):
        progression_logits = None
        (
            pooler_output,
            last_hidden_state,
            observation_det_logits,
            observation_cls_logits,
        ) = self.encode_image(
            input_pixels,
            require_logits=require_logits,
        )
        if temporal_mask.sum() > 0:
            (
                prior_pooler_output,
                prior_last_hidden_state,
                _,
                _,
            ) = self.encode_image(
                input_temporal_pixels,
                require_logits=False,
            )
        else:
            prior_pooler_output = torch.zeros_like(pooler_output)
            prior_last_hidden_state = torch.zeros_like(last_hidden_state)

        if require_logits:
            progression_pooler_output = torch.cat(
                (pooler_output, prior_pooler_output), dim=-1
            )
            progression_logits = self.progression_cls(progression_pooler_output)

        loss = None
        observation_loss = None
        progression_loss = None
        if observations is not None:
            observations_det = (observations != 2).float()
            observations_cls = (observations == 1).float()
            weight = (
                torch.ones_like(observations_det[:, :-1])
                + self.config.alpha * observations_det[:, :-1]
            )
            loss_fct = nn.BCEWithLogitsLoss(weight=weight.view(-1))
            loss = loss_fct(
                observation_det_logits.view(-1),
                observations_det[:, :-1].reshape(-1),
            )

            observation_cls_loss = self.bceloss_with_mask(
                observation_cls_logits,
                observations_cls,
                mask=observations_det,
            )
            loss = loss + observation_cls_loss
            observation_loss = loss
        if progressions is not None:
            num_label = progressions.size(-1)
            mask = temporal_mask.unsqueeze(-1).expand(-1, num_label)
            progression_loss = self.bceloss_with_mask(
                progression_logits, progressions.float(), mask
            )
            if loss is None:
                loss = progression_loss
            else:
                loss = loss + progression_loss
        return VisualOutput(
            loss=loss,
            observation_loss=observation_loss,
            progression_loss=progression_loss,
            observation_det_logits=observation_det_logits,
            observation_cls_logits=observation_cls_logits,
            progression_logits=progression_logits,
            last_hidden_state=(last_hidden_state, prior_last_hidden_state),
        )

    def bceloss_with_mask(self, logits, labels, mask, weight=None):
        # if weight is not None:
        #     weight = torch.ones_like(labels) + weight * labels
        loss_fct = nn.BCEWithLogitsLoss(reduction="none", weight=weight)
        loss = loss_fct(logits, labels)
        # norm = mask.sum()
        # norm = torch.max(norm, torch.ones_like(norm))
        # loss = (loss * mask).sum() / norm
        loss = (loss * mask).mean()
        return loss
