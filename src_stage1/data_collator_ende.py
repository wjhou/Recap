import random
import warnings
from dataclasses import dataclass
from optparse import Option
from re import A
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from PIL import Image
from regex import E
from transformers import DataCollatorForSeq2Seq

# from transformers.utils import PaddingStrategy
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

        if observations is not None:
            batch_outputs["observations"] = []
            for feature in features:
                batch_outputs["observations"].append(feature["observations"])

        if progressions is not None:
            batch_outputs["progressions"] = progressions

        features = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        features["input_pixels"] = torch.cat(input_pixels, dim=0)
        features["temporal_mask"] = torch.zeros((len(temporal_image_paths)))
        for i, image_path in enumerate(temporal_image_paths):
            if len(image_path) > 0:
                features["temporal_mask"][i] = 1
        features["input_temporal_pixels"] = torch.cat(input_temporal_pixels, dim=0)

        if report_ids is not None:
            features["report_ids"] = report_ids
            features["is_temporal"] = is_temporal
        return features

    def pad_sequence(self, seqs, padding_idx, max_len):
        new_seqs = []
        for seq in seqs:
            seq_len = len(seq)
            diff = max_len - seq_len
            new_seqs.append(seq + [padding_idx] * diff)
        return new_seqs
