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
        # deeper layers for representation fusion
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

    def encode_image(self, input_pixels):
        visual_outputs = self.visual_extractor(input_pixels)
        pooler_output = visual_outputs.pooler_output
        last_hidden_state = visual_outputs.last_hidden_state
        return (
            pooler_output,
            last_hidden_state,
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
        pooler_output, last_hidden_state = self.encode_image(input_pixels)
        current_pooler_output = pooler_output
        if temporal_mask.sum() > 0:
            prior_pooler_output, prior_last_hidden_state = self.encode_image(
                input_temporal_pixels
            )
        else:
            prior_pooler_output = torch.zeros_like(pooler_output)
            prior_last_hidden_state = torch.zeros_like(last_hidden_state)
        prior_pooler_output = prior_pooler_output * temporal_mask.unsqueeze(-1)
        prior_last_hidden_state = prior_last_hidden_state * temporal_mask.unsqueeze(
            -1
        ).unsqueeze(-1)
        observation_det_logits = None
        observation_cls_logits = None
        progression_logits = None
        if require_logits:
            observation_det_logits = self.observation_det(pooler_output)
            observation_cls_logits = self.observation_cls(pooler_output)
            pooler_output = torch.cat(
                (pooler_output, prior_pooler_output), dim=-1)
            progression_logits = self.progression_cls(pooler_output.detach())

        loss = None
        observation_loss = None
        progression_loss = None
        if require_logits and observations is not None:
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
        if require_logits and progressions is not None and self.config.beta > 0:
            num_label = progressions.size(-1)
            mask = temporal_mask.unsqueeze(-1).expand(-1, num_label)
            progression_loss = (
                self.bceloss_with_mask(
                    progression_logits, progressions.float(), mask)
                * self.config.beta
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
            pooler_output=current_pooler_output, 
        )

    def bceloss_with_mask(self, logits, labels, mask, weight=None):
        loss_fct = nn.BCEWithLogitsLoss(reduction="none", weight=weight)
        loss = loss_fct(logits, labels)
        loss = (loss * mask).mean()
        return loss
