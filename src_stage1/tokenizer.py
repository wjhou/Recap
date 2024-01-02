import copy
import json
import re
from collections import Counter, defaultdict

import pandas as pd
from transformers.tokenization_utils import PreTrainedTokenizer
import os
import pickle


class Tokenizer:
    def __init__(self, config):
        self.model_input_names = ["nodes"]
        self.padding_side = "right"
        self.ann_path = config.annotation_file
        self.threshold = config.threshold
        self.dataset = config.dataset
        if self.dataset == "iu_xray":
            self.clean_report = Tokenizer.clean_report_iu_xray
        else:
            self.clean_report = Tokenizer.clean_report_mimic_cxr
        print(self.clean_report)
        self.ann = json.loads(open(self.ann_path, "r").read())
        self.token2idx, self.idx2token, self.special_tokens = self.create_vocabulary()
        self.bos_token_id = self.eos_token_id = self.decoder_start_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 2

    def create_vocabulary(self):
        total_tokens = []
        for example in self.ann["train"]:
            tokens = self.clean_report(example["report"]).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold]
        vocab.sort()
        special_tokens = ["[CLS]", "[PAD]", "[UNK]"]
        vocab = special_tokens + vocab
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx
            idx2token[idx] = token
        return token2idx, idx2token, special_tokens[:-1]

    @staticmethod
    def clean_report_iu_xray(report):
        def report_cleaner(t):
            return (
                t.replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("1. ", "")
                .replace(". 2. ", ". ")
                .replace(". 3. ", ". ")
                .replace(". 4. ", ". ")
                .replace(". 5. ", ". ")
                .replace(" 2. ", ". ")
                .replace(" 3. ", ". ")
                .replace(" 4. ", ". ")
                .replace(" 5. ", ". ")
                .strip()
                .lower()
                .split(". ")
            )

        def sent_cleaner(t):
            return re.sub(
                "[.,?;*!%^&_+():-\[\]{}]",
                "",
                t.replace('"', "")
                .replace("/", "")
                .replace("\\", "")
                .replace("'", "")
                .strip()
                .lower(),
            )

        tokens = [
            sent_cleaner(sent).strip() + " ."
            for sent in report_cleaner(report)
            if len(sent_cleaner(sent).strip()) > 0
        ]
        report = " ".join(tokens)
        return report

    @staticmethod
    def clean_report_mimic_cxr(report):
        def report_cleaner(t):
            return (
                t.replace("\n", " ")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("1. ", "")
                .replace(". 2. ", ". ")
                .replace(". 3. ", ". ")
                .replace(". 4. ", ". ")
                .replace(". 5. ", ". ")
                .replace(" 2. ", ". ")
                .replace(" 3. ", ". ")
                .replace(" 4. ", ". ")
                .replace(" 5. ", ". ")
                .strip()
                .lower()
                .split(". ")
            )

        def sent_cleaner(t):
            return re.sub(
                "[.,?;*!%^&_+():-\[\]{}]",
                "",
                t.replace('"', "")
                .replace("/", "")
                .replace("\\", "")
                .replace("'", "")
                .lower()
                .strip(),
            )

        tokens = [
            sent_cleaner(sent).strip() + " ."
            for sent in report_cleaner(report)
            if len(sent_cleaner(sent).strip()) > 0
        ]
        report = " ".join(tokens)
        return report

    @staticmethod
    def load_tag2ids(
        tag_path,
        train_idxs=None,
        need_header=False,
    ):
        cached_path = tag_path + ".pkl"
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                tags = pickle.load(f)
        else:
            tags = pd.read_csv(tag_path)
            with open(cached_path, "wb") as f:
                pickle.dump(tags, file=f)
        # tags = tags.fillna(0).replace(-1, 1)
        # tags = tags.replace(-1, 1).replace(0, 1).fillna(0)
        tags = tags.replace(-1, 1).fillna(2)
        diseases = list(tags)[2:]
        id2tags = defaultdict(list)
        weight = [0] * len(diseases)
        count = 0
        for i in range(len(tags)):
            tag = tags.iloc[i]
            idx = tag[1]
            id2tags[idx] = list(tag[2:].values)
            # flip `No Finding`, negative stands for abnormal
            # id2tags[idx][-1] = 1 - id2tags[idx][-1]
            if train_idxs is not None and idx in train_idxs:
                weight = [w + v for w, v in zip(weight, id2tags[idx])]
                count += 1

        weight = [w / max(count, 1) for w in weight]
        if not need_header:
            return id2tags, weight
        else:
            return id2tags, diseases, weight

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx["[UNK]"]
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [self.decoder_start_token_id] + ids + [self.eos_token_id]
        return ids

    def encode(
        self,
        report,
        add_special_tokens=True,
    ):
        ids = []
        tokens = self.clean_report(report).split()
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        if add_special_tokens:
            ids = [self.decoder_start_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens=True, separator=" "):
        txt = []
        for i, idx in enumerate(ids):
            if idx not in self.idx2token:
                idx = self.unk_token_id
            token = self.idx2token[idx]
            if skip_special_tokens and token in self.special_tokens:
                continue
            txt.append(token)
        return separator.join(txt)

    def batch_decode(self, ids_batch, skip_special_tokens=True, separator=" "):
        out = []
        for ids in ids_batch:
            out.append(
                self.decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    separator=separator,
                )
            )
        return out

    def save_pretrained(self, save_directory):
        return ""

    def update_progression_tokens(self, observations, statuses):
        token_id = len(self.token2idx)
        for obs in observations:
            for status in statuses:
                token = f"[{obs}_{status}]"
                if token not in self.token2idx:
                    self.token2idx[token] = token_id
                    self.idx2token[token_id] = token
                    self.special_tokens.append(token)
                    token_id += 1
        self.token2idx["[PRO]"] = token_id
        self.idx2token[token_id] = "[PRO]"
        self.special_tokens.append("[PRO]")

    def search_progression_token(self, observation, status):
        return self.token2idx[f"[{observation}_{status}]"]


class TagTokenizer:
    def __init__(self, header) -> None:
        self.head2id = {head: idx for idx, head in enumerate(header)}

    def encode(self, tags):
        tag_ids = []
        for tag in tags:
            tag_ids.append(self.head2id[tag])
        if len(tag_ids) == 0:
            tag_ids = [len(self.head2id) + 1]
        return tag_ids
