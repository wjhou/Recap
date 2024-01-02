from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torchvision.models as models
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class VisualOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    observation_logits: torch.FloatTensor = None
    progression_logits: torch.FloatTensor = None


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
        visual_extractor,
    ):
        super().__init__(config)
        visual_extractor_name, d_visual = visual_extractor
        self.visual_extractor = VisualExtractor(visual_extractor_name)

        self.feature_space_transformation_nn = nn.Sequential(
            nn.Linear(in_features=d_visual, out_features=self.config.n_embd),
            nn.ReLU(),
            nn.Dropout(self.config.resid_pdrop),
        )
        self.observation_transformations = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_visual, self.config.n_embd),
                    nn.ReLU(),
                    nn.Dropout(self.config.resid_pdrop),
                )
                for _ in range(self.config.num_observation)
            ]
        )
        # self.observation_attn = nn.Linear(self.config.n_embd, self.config.num_observation)
        self.observation_cls = nn.Linear(d_visual, self.config.num_observation)
        self.progression_cls = nn.Sequential(
            nn.Linear(self.config.n_embd * 2, self.config.n_embd),
            nn.ReLU(),
            nn.Dropout(self.config.resid_pdrop),
            nn.Linear(self.config.n_embd, self.config.num_progression),
        )

    def encode_image(self, input_pixels, observations=None):
        image_hidden_states = self.visual_extractor(input_pixels)
        # observation_attn_weight = torch.softmax(
        #     self.observation_attn(image_hidden_states), dim=1
        # )
        # observation_hidden_states = observation_attn_weight.permute(0, 2, 1).bmm(
        #     image_hidden_states
        # )
        observation_logits = self.observation_cls(image_hidden_states.mean(dim=1))
        observation_hidden_states = torch.stack(
            [
                self.observation_transformations[i](image_hidden_states)
                for i in range(self.config.num_observation)
            ],
            dim=1,
        ).mean(dim=2)
        image_hidden_states = self.feature_space_transformation_nn(image_hidden_states)

        if observations is not None:
            observation_mask = observations
        else:
            observation_mask = (observation_logits > 0).float()
        attention_mask = torch.cat(
            (torch.ones_like(image_hidden_states[..., 0]), observation_mask), dim=-1
        )
        image_hidden_states = torch.cat(
            (image_hidden_states, observation_hidden_states), dim=1
        )
        return observation_hidden_states, attention_mask, observation_logits

    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        input_temporal_pixels: torch.FloatTensor = None,
        temporal_mask: torch.FloatTensor = None,
        observations: Optional[torch.FloatTensor] = None,
        progressions: Optional[torch.FloatTensor] = None,
    ):
        observation_logits = None
        progression_logits = None
        (
            obs_hidden_states,
            _,
            observation_logits,
        ) = self.encode_image(input_pixels, observations)
        (
            prior_obs_hidden_states,
            _,
            _,
        ) = self.encode_image(input_temporal_pixels, observations)

        # progression_hidden_states = torch.cat(
        #     (obs_hidden_states, prior_obs_hidden_states), dim=-1
        # )[:, :-2]
        # progression_logits = self.progression_cls(progression_hidden_states)

        loss = None
        if observations is not None:
            weight = torch.ones_like(observations) + self.config.alpha * observations
            loss_fct = nn.BCEWithLogitsLoss(weight=weight.view(-1))
            loss = loss_fct(
                observation_logits.view(-1),
                observations.view(-1),
            )

        if progressions is not None and False:
            loss_fct = nn.CrossEntropyLoss()
            progression_loss = loss_fct(
                progression_logits.view(-1, self.config.num_progression),
                progressions.view(-1),
            )
            loss = loss + progression_loss

        return VisualOutput(
            loss=loss,
            observation_logits=observation_logits,
            progression_logits=progression_logits,
        )
