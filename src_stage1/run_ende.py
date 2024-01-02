#!/usr/bin/env python
# coding=utf-8
import json
import logging
import os
import sys

import datasets
import torch
import transformers
from torchvision import transforms
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    BartConfig,
)
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer_utils import get_last_checkpoint
from data_arguments import DataTrainingArguments
from data_collator_ende import DataCollatorForEnDe as DataCollatorForSeq2Seq
from dataset_ende import DatasetCustom
from model_arguments import ModelArguments
from seq2seqtrainer_metrics_ende import Seq2SeqTrainerGenMetrics
from train_eval_ende_full import train
from transformers import ViTFeatureExtractor

from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings(
    action="ignore", category=UndefinedMetricWarning, module="sklearn"
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    Seq2SeqTrainer = Seq2SeqTrainerGenMetrics

    from tokenizer import Tokenizer

    data_args.dataset = (
        "mimic_abn" if "mimic_abn" in data_args.annotation_file else "mimic_cxr"
    )

    logger.info("***************************")
    logger.info("***************************")
    logger.info(data_args)
    logger.info("***************************")
    logger.info("***************************")

    logger.info("***************************")
    logger.info("***************************")
    logger.info(model_args)
    logger.info("***************************")
    logger.info("***************************")

    # load necessary data
    ref_annotation = None
    if data_args.miss_annotation_file is not None:
        with open(data_args.miss_annotation_file, "r", encoding="utf-8") as f:
            ref_annotation = json.load(f)
    with open(data_args.annotation_file, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    # temporal information
    with open(data_args.history, "r", encoding="utf-8") as f:
        temporal_ids = json.load(f)

    # ngram labels
    train_idxs = {sample["id"] for sample in annotation["train"]}
    # observation labels
    id2tags, observation_category, observation_weight = Tokenizer.load_tag2ids(
        data_args.chexbert_label,
        need_header=True,
        train_idxs=train_idxs,
    )


    from models.modeling_vit import VisualEncoder

    checkpoint = "GanjinZero/biobart-base"
    config = BartConfig.from_pretrained(checkpoint)
    config.num_observation = len(observation_category)
    config.num_progression = 3
    config.observation_category = observation_category
    config.alpha = data_args.alpha
    config.beta = data_args.beta
    config.observation_weight = observation_weight
    config.pretrained_visual_extractor = "google/vit-base-patch16-224-in21k"
    config.checkpoint = "google/vit-base-patch16-224-in21k"
    processor = ViTFeatureExtractor.from_pretrained(config.pretrained_visual_extractor)
    model = VisualEncoder(config=config)
    logger.info("***************************")
    logger.info("***** Model Structure *****")
    logger.info(model)
    logger.info("***************************")
    logger.info("***************************")
    train_dataset = eval_dataset = test_dataset = None

    if data_args.debug_model:
        debug_data_size = 16
        for key in temporal_ids:
            ref_ids = {report["id"] for report in ref_annotation[key]}
            subject_ids = list(temporal_ids[key].keys())[:debug_data_size]
            temporal_ids[key] = {
                subject_id: temporal_ids[key][subject_id] for subject_id in subject_ids
            }
            ids = set(subject_ids)
            annotation[key] = [
                ann
                for ann in annotation[key]
                if ann["id"] in ids
                and temporal_ids[key][ann["id"]]["object_id"] in ref_ids
            ]

    if training_args.do_train:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        )
        train_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            ref_annotation=ref_annotation,
            temporal_ids=temporal_ids,
            split="train",
            id2tags=id2tags,
            processor=processor,
            observation_category=observation_category,
            transform=transform,
        )
        eval_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            ref_annotation=ref_annotation,
            temporal_ids=temporal_ids,
            split="valid",
            id2tags=id2tags,
            processor=processor,
            observation_category=observation_category,
        )
    if training_args.do_predict:
        test_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            ref_annotation=ref_annotation,
            temporal_ids=temporal_ids,
            split="test",
            id2tags=id2tags,
            processor=processor,
            observation_category=observation_category,
        )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=None,
        model=model,
        padding=True,
        max_length=data_args.max_context_length,
        pad_to_multiple_of=8,
    )

    training_args.max_tgt_length = data_args.max_tgt_length
    training_args.num_beams = model_args.num_beams
    training_args.fast_lr = model_args.fast_lr
    training_args.remove_unused_columns = False
    data_args.max_steps = training_args.max_steps

    from transformers import EarlyStoppingCallback

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3 if data_args.dataset == "mimic_cxr" else 5,
            )
        ],
    )
    trainer.data_args = data_args
    if training_args.do_train:
        logger.info("*** Train ***")
        train(
            training_args,
            data_args,
            last_checkpoint,
            trainer,
            train_dataset,
        )

    # Prediction
    if training_args.do_predict:
        logger.info("*** Test ***")
        if model_args.test_model_name_or_path is not None:
            logger.info(
                "*** Test: Loading %s ***" % (model_args.test_model_name_or_path)
            )
            state_dict = torch.load(
                os.path.join(
                    model_args.test_model_name_or_path,
                    WEIGHTS_NAME,  # pytorch_model.bin
                ),
                map_location="cpu",
            )
            model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
        from train_eval_ende_full import eval_text

        print(model_args.num_beams)
        eval_text(
            max_tgt_length=data_args.max_tgt_length,
            model=model,
            test_dataset=trainer.get_test_dataloader(test_dataset),
            output_path=training_args.output_dir,
        )


if __name__ == "__main__":
    main()
