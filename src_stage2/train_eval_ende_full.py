import os

import torch
from tqdm import tqdm

from metrics import compute_scores
import json
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from collections import defaultdict
from radgraph import F1RadGraph
from src_stage2.models.modeling_bart import ViTBartForGeneration
from chexbert_eval import CONDITIONS


def pad_strings(strs):
    max_len = max([len(s) for s in strs])
    return [s + " " * (max_len - len(s)) for s in strs]


def train(training_args, data_args, last_checkpoint, trainer, train_dataset):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def eval_text(
    max_tgt_length: int,
    model: ViTBartForGeneration,
    tokenizer,
    test_dataset,
    output_path: str,
    result_file_name: str = "results.txt",
    reference_file_name: str = "references.txt",
    prediction_file_name: str = "predictions.txt",
    num_beams=None,
    compute_ce_metric=None,
    chexbert=None,
    bert_tokenizer=None,
    f1radgraph=None,
    decoder_tokenizer=None,
):
    model.eval()

    max_length = max_tgt_length
    print("******************")
    print("Text generation max length", max_length)
    print("******************")

    # for all report
    predictions = []
    multi_predictions = []
    references = []
    temporal_references = []
    pre_nodes = []
    ref_nodes = []
    report_ids = []

    # for temporal_report
    predictions_with_temporal = []
    references_with_temporal = []
    is_temporals = []
    test_progress = tqdm(
        test_dataset,
        desc="Evaluating Model (Report Generation)",
    )
    if num_beams is None:
        num_beams = 1

    print("******************")
    print("Beam Size", num_beams)
    print("******************")

    with torch.no_grad():
        for i, batch in enumerate(test_progress):
            max_length = max_tgt_length
            min_length = 2
            encoder_outputs = model.get_encoder()(
                input_pixels=batch["input_pixels"].cuda(),
                input_temporal_pixels=batch["input_temporal_pixels"].cuda(),
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                progression_input_ids=batch["progression_input_ids"].cuda(),
                progression_attention_mask=batch["progression_attention_mask"].cuda(),
                temporal_mask=batch["temporal_mask"].cuda(),
                matrix=batch["matrix"].cuda(),
                nodes=batch["nodes"].cuda(),
                node_mask=batch["node_mask"].cuda(),
            )

            model_inputs = {
                "attention_mask": batch["attention_mask"].cuda(),
                "temporal_mask": batch["temporal_mask"].cuda(),
                "input_pixels": batch["input_pixels"].cuda(),
                "node_mask": batch["node_mask"].cuda(),
                "gather_index": batch["gather_index"].cuda(),
                "matrix": batch["matrix"].cuda(),
                "nodes": batch["nodes"].cuda(),
                "num_beams": num_beams,
                "max_length": max_length,
                "min_length": min_length,
                "decoder_start_token_id": model.config.decoder_start_token_id,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "pad_token_id": model.config.pad_token_id,
                "early_stopping": True,
                "return_dict_in_generate": True,
                "encoder_outputs": encoder_outputs,
                "num_return_sequences": num_beams,
            }
            outputs = model.generate(**model_inputs)
            output_sequences = outputs["sequences"]
            multi_prediction = decoder_tokenizer.batch_decode(
                output_sequences.tolist(),
                skip_special_tokens=True,
            )
            prediction = [
                p for pi, p in enumerate(multi_prediction) if (pi % num_beams) == 0
            ]
            labels = batch["labels"].masked_fill(
                batch["labels"] == -100,
                tokenizer.pad_token_id,
            )
            reference = decoder_tokenizer.batch_decode(
                labels.tolist(),
                skip_special_tokens=True,
            )
            node = [
                {
                    decoder_tokenizer.id2entity[n_]
                    for n_ in n
                    if n_ != -100
                    and (
                        ":" not in decoder_tokenizer.id2entity[n_]
                        or "-" in decoder_tokenizer.id2entity[n_]
                    )
                }
                for n in batch["nodes"].tolist()
            ]
            selected_pre_node = []
            selected_ref_node = []
            for p, r, n in zip(prediction, reference, node):
                tokens = p.split()
                selected = []
                token2node = defaultdict(list)
                for a in n:
                    tok = a.split("-")[-1]
                    token2node[tok].append(a)
                for t in tokens:
                    if t in token2node:
                        selected.extend(token2node[t])
                selected_pre_node.append(list(set(selected)))

                tokens = r.split()
                selected = []
                for t in tokens:
                    if t in token2node:
                        selected.extend(token2node[t])
                selected_ref_node.append(list(set(selected)))
            if batch["progression_input_ids"] is not None:
                temporal_reference = decoder_tokenizer.batch_decode(
                    batch["progression_input_ids"].tolist(),
                    skip_special_tokens=True,
                )
            else:
                temporal_reference = ["Empty" for _ in range(len(prediction))]
            prediction = [z.strip() for z in prediction]
            reference = [z.strip() for z in reference]
            predictions.extend(prediction)
            references.extend(reference)
            temporal_references.extend(temporal_reference)
            pre_nodes.extend(selected_pre_node)
            ref_nodes.extend(selected_ref_node)
            report_ids.extend(batch["report_ids"])

            for pi in range(0, len(multi_prediction), num_beams):
                ps = [z.strip() for z in multi_prediction[pi : pi + num_beams]]
                multi_predictions.append(ps)

            predictions_with_temporal.extend(
                [
                    pre
                    for is_temporal, pre in zip(batch["is_temporal"], prediction)
                    if is_temporal
                ]
            )
            references_with_temporal.extend(
                [
                    ref
                    for is_temporal, ref in zip(batch["is_temporal"], reference)
                    if is_temporal
                ]
            )
            is_temporals.extend(batch["is_temporal"])
    assert len(references) == len(predictions), "Prediction Num != Reference Num"

    ce_scores = [0, 0, 0]
    with torch.no_grad():
        (
            _,
            _,
            ce_scores,
            temporal_ce_scores,
            macro_ce_scores,
            macro_temporal_ce_scores,
            tem_scores,
        ) = compute_ce_metric(
            references=references,
            hypotheses=predictions,
            is_temporals=is_temporals,
            chexbert=chexbert,
            bert_tokenizer=bert_tokenizer,
        )
        print("--------------------------------------------------------------")
        print(
            "Binary CE Score\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (ce_scores[0], ce_scores[1], ce_scores[2])
        )
        print(
            "Binray Temporal CE Score\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (temporal_ce_scores[0], temporal_ce_scores[1], temporal_ce_scores[2])
        )
        print(
            "Macro CE Score\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (macro_ce_scores[0], macro_ce_scores[1], macro_ce_scores[2])
        )
        print(
            "Macro Temporal CE Score\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (
                macro_temporal_ce_scores[0],
                macro_temporal_ce_scores[1],
                macro_temporal_ce_scores[2],
            )
        )
        print(
            "TEM Score\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (tem_scores[0], tem_scores[1], tem_scores[2])
        )
        print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    for i in range(5):
        print("Sample Prediction\t%d:" % i, predictions[i])
        print("Sample Reference\t%d:" % i, references[i])
    print("--------------------------------------------------------------")
    bleu_scores = compute_scores(
        gts={index: [gt] for index, gt in enumerate(references)},
        res={index: [re] for index, re in enumerate(predictions)},
    )
    for score in bleu_scores:
        print("%s\t%0.4f" % (score, bleu_scores[score]))
    bleu_scores_with_temporal = compute_scores(
        gts={index: [gt] for index, gt in enumerate(references_with_temporal)},
        res={index: [re] for index, re in enumerate(predictions_with_temporal)},
    )
    for score in bleu_scores_with_temporal:
        print("temporal_%s\t%0.4f" % (score, bleu_scores_with_temporal[score]))
    print("--------------------------------------------------------------")
    return bleu_scores
