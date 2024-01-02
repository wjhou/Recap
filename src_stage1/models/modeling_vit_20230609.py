from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torchvision.models as models
from transformers import PreTrainedModel, ViTModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class VisualOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    observation_logits: torch.FloatTensor = None
    progression_logits: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None


class VisualExtractor(nn.Module):
    def __init__(self, visual_extractor):
        super(VisualExtractor, self).__init__()
        model = getattr(models, visual_extractor)(pretrained=True)
        # num_fts = model.fc.in_features
        # model.fc = nn.Linear(num_fts, 512, bias=False)
        # medclip_state_dict = torch.load(
        #     "../CLIP/pretrained/medclip-resnet/clip_resnet50.bin"
        # )
        # model.load_state_dict(medclip_state_dict.state_dict())
        modules = list(model.children())
        self.model = nn.Sequential(*modules[:-2])

    def forward(self, images):
        patch_feats = self.model(images)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(
            batch_size,
            feat_size,
            -1,
        ).permute(0, 2, 1)
        return patch_feats


class VisualEncoder(PreTrainedModel):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.visual_extractor = ViTModel.from_pretrained(
            config.pretrained_visual_extractor
        )
        self.observation_cls = nn.Linear(
            self.visual_extractor.config.hidden_size, config.num_observation
        )
        self.progression_cls = nn.Linear(
            self.visual_extractor.config.hidden_size * 2, config.num_progression
        )

    def encode_image(self, input_pixels):
        visual_outputs = self.visual_extractor(input_pixels)
        pooler_output = visual_outputs.pooler_output
        last_hidden_state = visual_outputs.last_hidden_state
        observation_logits = self.observation_cls(pooler_output)
        return pooler_output, last_hidden_state, observation_logits

    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        input_temporal_pixels: torch.FloatTensor = None,
        temporal_mask: torch.FloatTensor = None,
        observations: Optional[torch.FloatTensor] = None,
        progressions: Optional[torch.FloatTensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        observation_logits = None
        progression_logits = None
        (
            pooler_output,
            last_hidden_state,
            observation_logits,
        ) = self.encode_image(input_pixels)
        (
            prior_pooler_output,
            prior_last_hidden_state,
            _,
        ) = self.encode_image(input_temporal_pixels)

        progression_pooler_output = torch.cat(
            (pooler_output, prior_pooler_output), dim=-1
        )
        progression_logits = self.progression_cls(progression_pooler_output)

        loss = None
        if observations is not None:
            weight = torch.ones_like(observations) + self.config.alpha * observations
            loss_fct = nn.BCEWithLogitsLoss(weight=weight.view(-1))
            loss = loss_fct(
                observation_logits.view(-1),
                observations.view(-1),
            )

        if progressions is not None:
            progression_loss = self.bceloss_with_mask(
                progression_logits, progressions.float(), temporal_mask
            )
            if loss is None:
                loss = progression_loss
            else:
                loss = loss + progression_loss

        return VisualOutput(
            loss=loss,
            observation_logits=observation_logits,
            progression_logits=progression_logits,
            last_hidden_state=(last_hidden_state, prior_last_hidden_state),
        )

    def bceloss_with_mask(self, logits, labels, mask, weight=None):
        if weight is not None:
            weight = torch.ones_like(labels) + weight * labels
        loss_fct = nn.BCEWithLogitsLoss(reduction="none", weight=weight)
        num_label = labels.size(-1)
        mask = mask.unsqueeze(-1).expand(-1, num_label)
        loss = loss_fct(logits, labels)
        norm = mask.sum()
        norm = torch.max(norm, torch.ones_like(norm))
        loss = (loss * mask).sum() / norm
        return loss
