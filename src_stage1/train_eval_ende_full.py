import os

import torch
from tqdm import tqdm

import json
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from collections import defaultdict


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


def index2label(observation_labels, progression_labels, inputs):
    outputs = []
    for input in inputs:
        output = []
        for i, j in enumerate(input):
            if j == -100:
                continue
            o = observation_labels[i]
            p = progression_labels[j]
            output.append(o + "_" + p)
        outputs.append(output)
    return outputs


def eval_text(
    max_tgt_length: int,
    model,
    test_dataset,
    output_path: str,
    result_file_name: str = "results.txt",
    reference_file_name: str = "references.txt",
    prediction_file_name: str = "predictions.txt",
):
    model.eval()

    max_length = max_tgt_length
    print("******************")
    print("Text generation max length", max_length)
    print("******************")

    # for all report
    predictions = []
    references = []
    report_ids = []
    test_progress = tqdm(
        test_dataset,
        desc="Evaluating Model (Report Generation)",
    )
    observation_labels = model.config.observation_category
    progression_labels = ["Better", "Worse", "No status change"]
    pad_observation_labels = pad_strings(observation_labels)
    pad_progression_labels = pad_strings(progression_labels)
    # progression_labels = defaultdict(str)
    # progression_labels.update({0: "Better", 1: "Worse", 2: "No status change"})
    observation_det_preds = []
    observation_cls_preds = []
    observation_trues = []
    progression_preds = []
    progression_trues = []
    temporal_mask = []
    with torch.no_grad():
        for i, batch in enumerate(test_progress):
            model_inputs = {
                "input_pixels": batch["input_pixels"].cuda(),
                "input_temporal_pixels": batch["input_temporal_pixels"].cuda(),
                "temporal_mask": batch["temporal_mask"].cuda(),
            }
            outputs = model(**model_inputs)
            observation_det_logits = outputs.observation_det_logits
            observation_cls_logits = outputs.observation_cls_logits
            observation_det_pred = (observation_det_logits > 0).float().cpu().numpy()
            observation_cls_pred = (observation_cls_logits > 0).float().cpu().numpy()
            observation_true = batch["observations"].cpu().numpy()
            observation_det_preds.append(observation_det_pred)
            observation_cls_preds.append(observation_cls_pred)
            observation_trues.append(observation_true)
            progression_logits = outputs.progression_logits
            if progression_logits is not None:
                progression_pred = (progression_logits > 0).float().cpu().numpy()
                progression_true = batch["progressions"].cpu().numpy()
                progression_preds.append(progression_pred)
                progression_trues.append(progression_true)
            temporal_mask.extend(batch["temporal_mask"].tolist())
            report_ids.extend(batch["report_ids"])
    observation_det_preds = np.concatenate(observation_det_preds, axis=0)
    observation_cls_preds = np.concatenate(observation_cls_preds, axis=0)
    observation_trues = np.concatenate(observation_trues, axis=0)
    progression_preds = np.concatenate(progression_preds, axis=0)
    progression_trues = np.concatenate(progression_trues, axis=0)

    num_observation = observation_cls_preds.shape[1]
    ce_scores = [0, 0, 0]
    observation_preds = []

    def get_pred(a, b):
        if a == 1 and b == 1:
            return 1
        elif a == 1 and b == 0:
            return 0
        else:
            return 2

    print("--------------------------------------------------------------")
    for i in range(num_observation):
        y_cls_pred = observation_cls_preds[:, i]
        if i == num_observation - 1:
            y_det_pred = np.ones_like(y_cls_pred)
        else:
            y_det_pred = observation_det_preds[:, i]

        y_pred = [1 if a == 1 and b == 1 else 0 for a, b in zip(y_det_pred, y_cls_pred)]
        y_true = (observation_trues[:, i] == 1) + 0.0
        observation_preds.append(
            [get_pred(a, b) for a, b in zip(y_det_pred, y_cls_pred)]
        )
        i_ce_score = precision_recall_fscore_support(
            y_pred=y_pred,
            y_true=y_true,
            pos_label=1,
            average="binary",
        )[:-1]
        print(
            "%s\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (pad_observation_labels[i], *i_ce_score)
        )
        ce_scores = [
            ce_scores[i] + i_ce_score[i] / len(pad_observation_labels)
            for i in range(len(ce_scores))
        ]
    observation_preds = np.stack(observation_preds, axis=1)
    print("--------------------------------------------------------------")
    print(
        "Abnormal CE Scores\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
        % (ce_scores[0], ce_scores[1], ce_scores[2])
    )
    target = ce_scores[-1]
    ce_scores = [0, 0, 0]
    print("--------------------------------------------------------------")
    for i in range(num_observation):
        y_cls_pred = observation_cls_preds[:, i]
        if i == num_observation - 1:
            y_det_pred = np.ones_like(y_cls_pred)
        else:
            y_det_pred = observation_det_preds[:, i]
        y_pred = [1 if a == 1 and b == 0 else 0 for a, b in zip(y_det_pred, y_cls_pred)]
        y_true = (observation_trues[:, i] == 0) + 0.0
        i_ce_score = precision_recall_fscore_support(
            y_pred=y_pred,
            y_true=y_true,
            pos_label=1,
            average="binary",
        )[:-1]
        # print(
        #     "%s\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
        #     % (pad_observation_labels[i], *i_ce_score)
        # )
        ce_scores = [
            ce_scores[i] + i_ce_score[i] / len(pad_observation_labels)
            for i in range(len(ce_scores))
        ]
    print(
        "Normal CE Scores\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
        % (ce_scores[0], ce_scores[1], ce_scores[2])
    )
    print("--------------------------------------------------------------")
    observation_det_trues = observation_trues != 2 + 0.0
    det_score = precision_recall_fscore_support(
        y_pred=observation_det_preds.reshape(-1),
        y_true=observation_det_trues[:, :-1].reshape(-1),
        pos_label=1,
        average="binary",
    )[:-1]
    print("%s\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f" % ("Detection", *det_score))
    print("--------------------------------------------------------------")
    if len(progression_preds) > 0:
        num_progression = progression_trues.shape[1]
        pro_scores = [0, 0, 0]
        for i in range(num_progression):
            i_pro_score = precision_recall_fscore_support(
                y_pred=[
                    a for a, b in zip(progression_preds[:, i], temporal_mask) if b == 1
                ],
                y_true=[
                    a for a, b in zip(progression_trues[:, i], temporal_mask) if b == 1
                ],
                pos_label=1,
                average="binary",
            )[:-1]
            print(
                "%s\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
                % (pad_progression_labels[i], *i_pro_score)
            )
            pro_scores = [
                pro_scores[i] + i_pro_score[i] / len(pad_progression_labels)
                for i in range(len(pro_scores))
            ]
        print("--------------------------------------------------------------")
        print(
            "Pro Scores\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (pro_scores[0], pro_scores[1], pro_scores[2])
        )
        print("--------------------------------------------------------------")

    output_data = {}
    for (
        sample_index,
        report_id,
        obs_pred,
        obs_true,
        tm,
    ) in zip(
        range(len(report_ids)),
        report_ids,
        observation_preds,
        observation_trues,
        temporal_mask,
    ):
        hyp = [
            a + (":Positive" if b == 1 else ":Negative")
            for a, b in zip(observation_labels, obs_pred)
            if b == 1 or b == 0
        ]
        ref = [
            a + (":Positive" if b == 1 else ":Negative")
            for a, b in zip(observation_labels, obs_true)
            if b == 1 or b == 0
        ]
        output_data[report_id] = {
            "obs_hyp": hyp,
            "obs_ref": ref,
            # "pro_hyp": pro_pred,
            # "pro_ref": pro_true,
        }
        if len(progression_preds) > 0 and tm == 1:
            output_data[report_id]["pro_hyp"] = [
                a
                for a, b in zip(progression_labels, progression_preds[sample_index])
                if b == 1
            ]
            output_data[report_id]["pro_ref"] = [
                a
                for a, b in zip(progression_labels, progression_trues[sample_index])
                if b == 1
            ]

    if output_path:
        with open(
            os.path.join(output_path, result_file_name),
            "w",
            encoding="utf-8",
        ) as f, open(
            os.path.join(output_path, reference_file_name),
            "w",
            encoding="utf-8",
        ) as f2, open(
            os.path.join(output_path, prediction_file_name),
            "w",
            encoding="utf-8",
        ) as f3:
            f.write(",".join(("Reference", "Prediction")) + "\n")
            for idx, pre, ref in zip(
                range(len(predictions)),
                predictions,
                references,
            ):
                f.write("Reference:\t%s\n" % ",".join(ref))
                f.write("Prediction:\t%s\n" % ",".join(pre))
                f3.write(pre + "\n")
                f2.write(ref + "\n")
            f.write("****************\n")
        with open(
            os.path.join(output_path, result_file_name.replace("txt", "json")),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
    return {"eval_BLEU_4": target}
