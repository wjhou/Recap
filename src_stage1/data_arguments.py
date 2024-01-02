from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    image_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    annotation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    miss_annotation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    history: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    graph_version: Optional[str] = field(
        default=None,
    )
    progression_graph: Optional[str] = field(
        default=None,
    )
    chexbert_label: Optional[str] = field(default=None)
    debug_model: Optional[bool] = field(default=False)
    max_tgt_length: Optional[int] = field(
        default=64,
    )
    is_stage1_pretrained: int = field(default=1)
    is_temporal: int = field(default=1)
    eval_on_gen: Optional[bool] = field(default=False)
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )
    alpha: Optional[float] = field(default=3)
    beta: Optional[float] = field(default=3)
    wo_op: Optional[int] = field(default=1)
    wo_obs: Optional[int] = field(default=1)
    wo_pro: Optional[int] = field(default=1)
    wo_prr: Optional[int] = field(default=1)
    topk: Optional[int] = field(default=10)
    lambda_: Optional[float] = field(default=0.5)
