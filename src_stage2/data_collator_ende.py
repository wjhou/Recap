from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import DataCollatorForSeq2Seq

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


@dataclass
class DataCollatorForEnDe(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        input_ids = (
            [feature["input_ids"] for feature in features]
            if "input_ids" in features[0].keys()
            else None
        )
        progression_input_ids = (
            [feature["progression_input_ids"] for feature in features]
            if "progression_input_ids" in features[0].keys()
            else None
        )
        matrix = (
            [feature["matrix"] for feature in features]
            if "matrix" in features[0].keys()
            else None
        )
        report_ids = (
            [feature["report_ids"] for feature in features]
            if "report_ids" in features[0].keys()
            else None
        )
        is_temporal = (
            [feature["is_temporal"] for feature in features]
            if "is_temporal" in features[0].keys()
            else None
        )
        observations = (
            [feature["observations"] for feature in features]
            if "observations" in features[0].keys()
            else None
        )
        prior_observations = (
            [feature["prior_observations"] for feature in features]
            if "prior_observations" in features[0].keys()
            else None
        )
        prior_entity_ids = (
            [feature["prior_entity_ids"] for feature in features]
            if "prior_entity_ids" in features[0].keys()
            else None
        )
        temporal_image_paths = (
            [feature["temporal_image_path"] for feature in features]
            if "temporal_image_path" in features[0].keys()
            else None
        )
        progressions = (
            [feature["progressions"] for feature in features]
            if "progressions" in features[0].keys()
            else None
        )
        input_pixels = (
            [feature["input_pixels"] for feature in features]
            if "input_pixels" in features[0].keys()
            else None
        )
        input_temporal_pixels = (
            [feature["input_temporal_pixels"] for feature in features]
            if "input_temporal_pixels" in features[0].keys()
            else None
        )
        batch_outputs = {}

        if labels is not None:
            batch_outputs["labels"] = []
            batch_outputs["gate_labels"] = []
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                feature["labels"] = feature["labels"] + remainder
                feature["gate_labels"] = feature["gate_labels"] + remainder
                batch_outputs["labels"].append(feature["labels"])
                batch_outputs["gate_labels"].append(feature["gate_labels"])

        if input_ids is not None:
            batch_outputs["input_ids"] = []
            max_length = max(len(l) for l in input_ids)
            if self.pad_to_multiple_of is not None:
                max_length = (
                    (max_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_length - len(feature["input_ids"])
                )
                feature["input_ids"] = feature["input_ids"] + remainder
                batch_outputs["input_ids"].append(feature["input_ids"])

        if progression_input_ids is not None:
            batch_outputs["progression_input_ids"] = []
            max_length = max(len(l) for l in progression_input_ids)
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_length - len(feature["progression_input_ids"])
                )
                feature["progression_input_ids"] = (
                    feature["progression_input_ids"] + remainder
                )
                batch_outputs["progression_input_ids"].append(
                    feature["progression_input_ids"]
                )

        if observations is not None:
            batch_outputs["observations"] = []
            for feature in features:
                batch_outputs["observations"].append(feature["observations"])

        if matrix is not None:
            batch_outputs["node_mask"] = []
            batch_outputs["nodes"] = []
            batch_outputs["gather_index"] = []
            max_length = max(m.shape[-1] for m in matrix)
            for feature in features:
                batch_outputs["gather_index"].append(feature["gather_index"])

            for i, m in enumerate(matrix):
                feature = features[i]
                diff = max_length - m.shape[-1]
                m = np.pad(
                    m,
                    ((0, 0), (0, diff), (0, diff)),
                    mode="constant",
                    constant_values=0,
                )
                feature["node_mask"] = feature["node_mask"] + [0] * diff
                feature["nodes"] = feature["nodes"] + [-100] * diff
                matrix[i] = m
                batch_outputs["node_mask"].append(feature["node_mask"])
                batch_outputs["nodes"].append(feature["nodes"])

        if progressions is not None:
            batch_outputs["progressions"] = progressions
        features = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        features["input_pixels"] = torch.cat(input_pixels, dim=0)
        features["temporal_mask"] = torch.zeros((len(temporal_image_paths)))
        for i, image_path in enumerate(temporal_image_paths):
            if len(image_path) > 0:
                features["temporal_mask"][i] = 1
        features["input_temporal_pixels"] = torch.cat(
            input_temporal_pixels, dim=0)
        features["matrix"] = torch.from_numpy(np.stack(matrix, axis=0)).float()
        features["attention_mask"] = torch.ones_like(features["input_ids"]).masked_fill(
            features["input_ids"] == self.tokenizer.pad_token_id, 0
        )
        features["progression_attention_mask"] = torch.ones_like(
            features["progression_input_ids"]
        ).masked_fill(
            features["progression_input_ids"] == self.tokenizer.pad_token_id, 0
        )

        if report_ids is not None:
            features["report_ids"] = report_ids
            features["is_temporal"] = is_temporal
            features["prior_observations"] = prior_observations
            features["prior_entity_ids"] = prior_entity_ids

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids
        return features

    def pad_sequence(self, seqs, padding_idx, max_len):
        new_seqs = []
        for seq in seqs:
            seq_len = len(seq)
            diff = max_len - seq_len
            new_seqs.append(seq + [padding_idx] * diff)
        return new_seqs
