#!/usr/bin/env python
# coding=utf-8
import torch
from collections import OrderedDict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm

CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def load_chexbert(checkpoint_path):
    import sys

    sys.path.append("../CheXbert/src/")
    from models.bert_labeler import bert_labeler

    chexbert = bert_labeler()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model_state_dict"].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    chexbert.load_state_dict(new_state_dict, strict=False)
    print("Loaded reward model from {}".format(checkpoint_path))
    chexbert.eval()
    return chexbert.cuda()


def compute_ce_metric(references, hypotheses, is_temporals, chexbert, bert_tokenizer):
    def pad_strings(strs):
        max_len = max([len(s) for s in strs])
        return [s + " " * (max_len - len(s)) for s in strs]

    chexbert.eval()
    # CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}
    CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Positive"}
    NO_FINDING_CLASS_MAPPING = {0: "Negative", 1: "Positive"}
    LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 1}
    TEM_keywords = {
        "bigger",
        "change",
        "cleared",
        "constant",
        "decrease",
        "decreased",
        "decreasing",
        "elevated",
        "elevation",
        "enlarged",
        "enlargement",
        "enlarging",
        "expanded",
        "greater",
        "growing",
        "improved",
        "improvement",
        "improving",
        "increase",
        "increased",
        "increasing",
        "larger",
        "new",
        "persistence",
        "persistent",
        "persisting",
        "progression",
        "progressive",
        "reduced",
        "removal",
        "resolution",
        "resolved",
        "resolving",
        "smaller",
        "stability",
        "stable",
        "stably",
        "unchanged",
        "unfolded",
        "worse",
        "worsen",
        "worsened",
        "worsening",
        "unaltered",
    }
    batch_size = 128
    ref_observations = []
    hyp_observations = []
    y_preds = []
    y_trues = []
    macro_y_preds = []
    macro_y_trues = []
    for i in tqdm(range(0, len(references), batch_size), desc="Calculating CE Scores"):
        ref = references[i : i + batch_size]
        hyp = hypotheses[i : i + batch_size]
        ref_input = bert_tokenizer.batch_encode_plus(
            ref, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        hyp_input = bert_tokenizer.batch_encode_plus(
            hyp, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        ref_input = {k: v.cuda() for k, v in ref_input.items()}
        hyp_input = {k: v.cuda() for k, v in hyp_input.items()}
        ref_logits = chexbert(
            source_padded=ref_input["input_ids"],
            attention_mask=ref_input["attention_mask"],
        )
        hyp_logits = chexbert(
            source_padded=hyp_input["input_ids"],
            attention_mask=hyp_input["attention_mask"],
        )
        ref_status = [l.argmax(dim=1).tolist() for l in ref_logits]
        hyp_status = [l.argmax(dim=1).tolist() for l in hyp_logits]
        y_pred = np.zeros((len(ref_status[0]), len(CONDITIONS)))
        y_true = np.zeros((len(hyp_status[0]), len(CONDITIONS)))
        macro_y_pred = np.zeros((len(ref_status[0]), len(CONDITIONS)))
        macro_y_true = np.zeros((len(hyp_status[0]), len(CONDITIONS)))
        ref_obs = [[] for _ in range(len(ref_status[0]))]
        hyp_obs = [[] for _ in range(len(hyp_status[0]))]
        for i, c in enumerate(CONDITIONS):
            i_ref_status = ref_status[i]
            i_hyp_status = hyp_status[i]
            if c == "No Finding":
                class_mapping = NO_FINDING_CLASS_MAPPING
            else:
                class_mapping = CLASS_MAPPING
            for j in range(len(i_hyp_status)):  # batch_size
                macro_y_pred[j][i] = i_hyp_status[j]
                macro_y_true[j][i] = i_ref_status[j]
                if LABEL_MAPPING[i_hyp_status[j]] == 1:
                    y_pred[j][i] = 1
                if LABEL_MAPPING[i_ref_status[j]] == 1:
                    y_true[j][i] = 1
                if i_hyp_status[j] != 0 or c == "No Finding":
                    hyp_obs[j].append(":".join((c, class_mapping[i_hyp_status[j]])))
                if i_ref_status[j] != 0 or c == "No Finding":
                    ref_obs[j].append(":".join((c, class_mapping[i_ref_status[j]])))

        y_preds.append(y_pred)
        y_trues.append(y_true)
        macro_y_preds.append(macro_y_pred)
        macro_y_trues.append(macro_y_true)
        ref_observations.extend(ref_obs)
        hyp_observations.extend(hyp_obs)
    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    macro_y_preds = np.concatenate(macro_y_preds, axis=0)
    macro_y_trues = np.concatenate(macro_y_trues, axis=0)
    ce_prf = [0, 0, 0]
    macro_ce_prf = [0, 0, 0]
    temporal_ce_prf = [0, 0, 0]
    macro_temporal_ce_prf = [0, 0, 0]

    print("--------------------------------------------------------------")
    pad_conditions = pad_strings(CONDITIONS)
    for i, c in enumerate(CONDITIONS):
        # for all reports
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        i_prf = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average="binary", pos_label=1
        )
        ce_prf = [ce_prf[j] + i_prf[j] for j in range(3)]

        print(
            "%s\tPrec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (pad_conditions[i], i_prf[0], i_prf[1], i_prf[2])
        )

        y_true = macro_y_trues[:, i]
        y_pred = macro_y_preds[:, i]
        i_prf = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average="macro"
        )
        macro_ce_prf = [macro_ce_prf[j] + i_prf[j] for j in range(3)]

        # for reports with temporal information
        y_true = [z for z, k in zip(y_trues[:, i], is_temporals) if k]
        y_pred = [z for z, k in zip(y_preds[:, i], is_temporals) if k]
        i_prf = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average="binary", pos_label=1
        )
        temporal_ce_prf = [temporal_ce_prf[j] + i_prf[j] for j in range(3)]

        y_true = [z for z, k in zip(macro_y_trues[:, i], is_temporals) if k]
        y_pred = [z for z, k in zip(macro_y_preds[:, i], is_temporals) if k]
        i_prf = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average="macro"
        )
        macro_temporal_ce_prf = [macro_temporal_ce_prf[j] + i_prf[j] for j in range(3)]
    print("--------------------------------------------------------------")
    ce_prf = [ce_prf[j] / len(CONDITIONS) for j in range(3)]
    macro_ce_prf = [macro_ce_prf[j] / len(CONDITIONS) for j in range(3)]
    temporal_ce_prf = [temporal_ce_prf[j] / len(CONDITIONS) for j in range(3)]
    macro_temporal_ce_prf = [
        macro_temporal_ce_prf[j] / len(CONDITIONS) for j in range(3)
    ]

    tp = 0
    count_gen = 0
    count_ref = 0
    for ref, hyp, is_temporal in zip(references, hypotheses, is_temporals):
        if not is_temporal:
            continue
        ref_tem = set([z for z in ref.split() if z in TEM_keywords])
        hyp_tem = set([z for z in hyp.split() if z in TEM_keywords])
        tp += len(ref_tem & hyp_tem)
        count_gen += len(hyp_tem)
        count_ref += len(ref_tem)
    tem_prec = tp / max(count_gen, 1)
    tem_rec = tp / max(count_ref, 1)
    tem_f1 = 2 * tem_prec * tem_rec / max((tem_prec + tem_rec), 0.1)
    tem_score = [tem_prec, tem_rec, tem_f1]
    return (
        ref_observations,
        hyp_observations,
        ce_prf,
        temporal_ce_prf,
        macro_ce_prf,
        macro_temporal_ce_prf,
        tem_score,
    )
