#!/usr/bin/env python
# coding=utf-8
import json
import logging
import os
import sys

import datasets
import torch
from torchvision import transforms
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    BertTokenizer,
    BartTokenizer,
    BartConfig,
)
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer_utils import get_last_checkpoint
from radgraph import F1RadGraph
from data_collator_ende import DataCollatorForEnDe as DataCollatorForSeq2Seq
from dataset_ende import DatasetCustom
from model_arguments import ModelArguments
from seq2seqtrainer_metrics_ende import Seq2SeqTrainerGenMetrics
from train_eval_ende_full import train
from transformers import ViTFeatureExtractor
from chexbert_eval import compute_ce_metric, load_chexbert, build_progression_graph
import copy
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from src_stage2.models.modeling_bart import ViTBartForGeneration

sys.path.append("../")
from src_stage1.data_arguments import DataTrainingArguments

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

    data_args.threshold = 3 if data_args.dataset == "mimic_abn" else 10
    # ngram labels
    train_idxs = {sample["id"] for sample in annotation["train"]}
    # observation labels
    id2tags, observation_category, observation_weight = Tokenizer.load_tag2ids(
        data_args.chexbert_label,
        need_header=True,
        train_idxs=train_idxs,
    )
    checkpoint = "GanjinZero/biobart-base"
    bart_tokenizer = BartTokenizer.from_pretrained(checkpoint)
    tokenizer = Tokenizer(data_args, observation_category)

    progression_graph = build_progression_graph(
        progression_triples=json.load(
            open(data_args.progression_graph, "r", encoding="utf-8")
        ),
        observations=observation_category,
        topk_entity=data_args.topk,
        tokenizer=tokenizer,
    )
    tokenizer.id2entity = progression_graph["id2entity"]
    chexbert = load_chexbert(model_args.chexbert_model_name_or_path)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    f1radgraph = F1RadGraph(reward_level="partial")

    config = BartConfig.from_pretrained(checkpoint)
    config.num_observation = len(observation_category)
    config.num_progression = 3
    config.num_rgcnlayer = 3
    config.num_relation = len(progression_graph["relation2id"])
    # config.num_entity = len(progression_graph["entity2id"])
    config.num_node = len(progression_graph["entity2id"])
    config.observation_category = observation_category
    config.alpha = data_args.alpha
    config.beta = data_args.beta
    config.observation_weight = observation_weight
    config.pretrained_visual_extractor = "google/vit-base-patch16-224-in21k"
    config.topk = data_args.topk
    processor = ViTFeatureExtractor.from_pretrained(config.pretrained_visual_extractor)

    config.add_cross_attention = True
    
    config.is_temporal = 1
    config.is_stage1_pretrained = int(data_args.is_stage1_pretrained)

    config.stage1_model_name_or_path = model_args.stage1_model_name_or_path
    if int(data_args.is_stage1_pretrained) == 0:
        config.stage1_model_name_or_path = None
    config.decoder_model_name_or_path = checkpoint
    config.num_path = 16 * 16 + 1
    config.lambda_ = data_args.lambda_
    config.id2entity = progression_graph["id2entity"]
    encoder_config = config
    decoder_config = copy.deepcopy(config)

    decoder_config.vocab_size = len(tokenizer.token2idx)
    decoder_config.decoder_layers = 3
    decoder_config.d_model = 768
    decoder_config.decoder_ffn_dim = 768
    decoder_config.decoder_attention_heads = 8
    decoder_config.encoder_layers = 3
    decoder_config.d_model = 768
    decoder_config.encoder_ffn_dim = 768
    decoder_config.encoder_attention_heads = 8
    decoder_config.activation_function = "relu"
    decoder_config.decoder_start_token_id = tokenizer.bos_token_id
    decoder_config.eos_token_id = tokenizer.eos_token_id
    decoder_config.bos_token_id = tokenizer.bos_token_id
    decoder_config.decoder_start_token_id = tokenizer.bos_token_id
    decoder_config.pad_token_id = tokenizer.pad_token_id
    data_args.vocab_size = decoder_config.vocab_size
    model = ViTBartForGeneration(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    model.observation_category = observation_category
    model.id2entity = progression_graph["id2entity"]
    data_args.vocab_size = len(tokenizer.token2idx)
    data_args.stage1_model_name_or_path = model_args.stage1_model_name_or_path
    data_args.stage1_eval_file = model_args.stage1_eval_file

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
        transform = None
        train_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            ref_annotation=ref_annotation,
            temporal_ids=temporal_ids,
            split="train",
            id2tags=id2tags,
            processor=processor,
            text_tokenizer=bart_tokenizer,
            tokenizer=tokenizer,
            progression_graph=progression_graph,
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
            text_tokenizer=bart_tokenizer,
            tokenizer=tokenizer,
            progression_graph=progression_graph,
            observation_category=observation_category,
            transform=None,
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
            text_tokenizer=bart_tokenizer,
            tokenizer=tokenizer,
            progression_graph=progression_graph,
            observation_category=observation_category,
            transform=None,
        )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
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
                early_stopping_patience=5 if data_args.dataset == "mimic_cxr" else 3,
            )
        ],
    )
    trainer.data_args = data_args
    trainer.chexbert = chexbert
    trainer.bert_tokenizer = bert_tokenizer
    trainer.f1radgraph = f1radgraph
    trainer.compute_ce_metric = compute_ce_metric
    trainer.tokenizer = bart_tokenizer
    trainer.decoder_tokenizer = tokenizer

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
            num_beams=model_args.num_beams,
            compute_ce_metric=compute_ce_metric,
            chexbert=chexbert,
            bert_tokenizer=bert_tokenizer,
            f1radgraph=f1radgraph,
            tokenizer=bart_tokenizer,
            decoder_tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
