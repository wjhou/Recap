import json
import os
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset
import torch
from data_arguments import DataTrainingArguments
from data_process_ende import process_examples
from tqdm import tqdm
from PIL import Image
import random


def load_images(root_path, image_paths):
    images = {}
    for image_path in tqdm(image_paths, desc="Loading Images"):
        for img_path in image_path:
            img_path = os.path.join(root_path, img_path)
            image = Image.open(img_path).convert("RGB")
            images[img_path] = image
    return images


def extract_temporal_info(
    samples,
    ref_samples,
    temporal_ids,
    entity2id=None,
    entity_label=None,
):
    id2sample = {sample["id"]: sample for sample in samples}
    if ref_samples is not None:
        ref_id2sample = {sample["id"]: sample for sample in ref_samples}
        for subject_id in temporal_ids:
            object_id = temporal_ids[subject_id]["object_id"]
            if object_id not in id2sample:
                id2sample[object_id] = ref_id2sample[object_id]

    for sample in samples:
        sample["temporal_image_path"] = []
        sample["temporal_entity"] = set()
        sample["current_entity"] = set()
        sample["temporal_predicate"] = []
        sample["temporal_id"] = None
        sample["temporal_report"] = ""

    for subject_id in tqdm(temporal_ids, desc="Updating Temooral Info"):
        predicate_object = temporal_ids[subject_id]
        predicate = predicate_object["predicate"]
        subject_example = id2sample[subject_id]

        object_id = predicate_object["object_id"]
        if object_id not in id2sample:
            print(object_id, "Not Found")
        else:
            object_example = id2sample[object_id]
            subject_example["temporal_image_path"] = object_example["image_path"]
            subject_example["temporal_report"] = object_example["report"]
            if object_id in entity_label:
                for e in entity_label[object_id]:
                    subject_example["temporal_entity"].add(e)
            subject_example["temporal_predicate"] = predicate
    return samples


class DatasetCustom(Dataset):
    def __init__(
        self,
        data_args: DataTrainingArguments,
        annotation,
        ref_annotation,
        temporal_ids,
        split: str,
        text_tokenizer,
        tokenizer,
        id2tags,
        processor,
        progression_graph,
        observation_category,
        transform=None,
        keep_columns={
            "id",
            "report",
            "image_path",
            "temporal_image_path",
            "temporal_entity",
            "current_entity",
            "temporal_predicate",
            "temporal_report",
        },
    ) -> None:
        super().__init__()
        self.text_tokenizer = text_tokenizer
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = data_args
        self.split = split
        self.dataset = data_args.dataset
        self.id2tags = id2tags
        examples = {kc: [] for kc in keep_columns}
        samples = annotation[split.replace("valid", "val")]
        self.temporal_ids = temporal_ids[split.replace("valid", "val")]
        ref_samples = None
        if ref_annotation is not None:
            ref_samples = ref_annotation[split.replace("valid", "val")]
        self.temporal_collection = temporal_ids.keys()
        (
            self.triples,
            self.entity2id,
            self.id2entity,
            self.relation2id,
            self.id2relation,
            self.entity2subid,
        ) = (
            progression_graph["triples"],
            progression_graph["entity2id"],
            progression_graph["id2entity"],
            progression_graph["relation2id"],
            progression_graph["id2relation"],
            progression_graph["entity2subid"],
        )
        with open(
            f"./data/{data_args.graph_version}/%s/id2entity.json" % data_args.dataset,
            "r",
            encoding="utf-8",
        ) as f:
            self.id2entity_label = json.load(f)

        samples = extract_temporal_info(
            samples,
            ref_samples,
            self.temporal_ids,
            self.entity2id,
            self.id2entity_label,
        )
        for sample in samples:
            for key in sample:
                if key not in examples:
                    continue
                examples[key].append(sample[key])
        for key in examples:
            print(key, examples[key][:3])
        (
            idxs,
            image_paths,
            temporal_image_paths,
            temporal_entity_ids,
            current_entity_ids,
            temporal_predicates,
            temporal_reports,
            labels,
        ) = process_examples(
            examples=examples,
            max_tgt_length=data_args.max_tgt_length,
            tokenizer=tokenizer,
        )
        self.data = [
            {
                "id": a,
                "image_path": b,
                "temporal_image_path": c,
                "temporal_entity_ids": d,
                "current_entity_ids": e,
                "temporal_predicates": f,
                "temporal_report": g,
                "labels": h,
            }
            for a, b, c, d, e, f, g, h in zip(
                idxs,
                image_paths,
                temporal_image_paths,
                temporal_entity_ids,
                current_entity_ids,
                temporal_predicates,
                temporal_reports,
                labels,
            )
        ]
        self.all_index = list(range(len(self.data)))

        self.observation2id = {obs: idx for idx, obs in enumerate(observation_category)}
        self.observation_category = observation_category
        self.transform = transform
        self.tokenizer = tokenizer
        self.triples_ = defaultdict(list)
        for (hid, rid), tids in self.triples.items():
            for i, tid in enumerate(tids):
                self.triples_[(hid, tid)].append(rid)

        if self.split != "train":
            path = data_args.stage1_model_name_or_path
            if self.split == "test":
                path = path + "results.json"
            else:
                path = path + data_args.stage1_eval_file
            self.op_data = json.load(open(path, "r", encoding="utf-8"))

    def __getitem__(self, index):
        idx = self.data[index]["id"]
        labels = self.data[index]["labels"]
        status2id = {"Better": 0, "Worse": 1, "No status change": 2}
        progressions = [0, 0, 0]
        for progression in self.data[index]["temporal_predicates"]:
            staid = status2id[progression]
            progressions[staid] = 1

        # current radiograph
        image_path = [
            os.path.join(self.data_args.image_path, a)
            for a in self.data[index]["image_path"]
        ]
        pixel_value = []
        for img_path in image_path:
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            image = self.processor(images=image, return_tensors="pt")["pixel_values"]
            pixel_value.append(image)
        pixel_value = torch.cat(pixel_value, dim=0)

        # prior radiograph
        temporal_image_path = [
            os.path.join(self.data_args.image_path, a)
            for a in self.data[index]["temporal_image_path"]
        ]
        pixel_value_temporal = torch.zeros_like(pixel_value)
        for i, img_path in enumerate(temporal_image_path):
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            image = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
            pixel_value_temporal[i] = image

        # load current observations
        id2nodelabel = {0: ":Negative", 1: ":Positive"}
        if self.split == "train":
            current_observations = [
                self.observation_category[pos] + id2nodelabel[tag]
                for pos, tag in enumerate(self.id2tags[idx])
                if tag != 2
            ]
        else:
            obs = self.op_data[idx]["obs_hyp"]
            current_observations = obs
        current_observations = sorted(
            current_observations,
            key=lambda x: self.observation_category.index(x.split(":")[0]),
        )

        # load prior observations
        prior_observations = []
        if (
            idx in self.temporal_ids
            and self.temporal_ids[idx]["object_id"] in self.id2tags
        ):
            prior_observations = [
                "pre_" + self.observation_category[pos] + id2nodelabel[tag]
                for pos, tag in enumerate(
                    self.id2tags[self.temporal_ids[idx]["object_id"]]
                )
                if tag != 2
            ]
        observation_prompt_ids = []
        report_prompt_ids = []

        if self.split == "train":
            observation_prompt_ids = [
                self.tokenizer.token2idx[
                    "[{}]".format(self.observation_category[pos] + id2nodelabel[tag])
                ]
                for pos, tag in enumerate(self.id2tags[idx])
                if tag != 2
            ]
        else:
            observation_prompt_ids = [
                self.tokenizer.token2idx["[{}]".format(o)]
                for o in sorted(
                    self.op_data[idx]["obs_hyp"],
                    key=lambda x: self.observation_category.index(x.split(":")[0]),
                )
            ]

        # load prior report
        if len(temporal_image_path) > 0:
            report_prompt_ids = self.data[index]["temporal_report"]

        # insert [FiV] or [FoV] to distinguish first visits and follow-up visits
        f_v = self.tokenizer.token2idx[
            "[First-Visit]" if len(temporal_image_path) == 0 else "[Follow-Up-Visit]"
        ]
        observation_prompt_ids = [f_v] + observation_prompt_ids
        input_ids = observation_prompt_ids
        size = len(input_ids)
        progression_input_ids = report_prompt_ids

        if size == 0:
            input_ids = [self.tokenizer.pad_token_id]
        if len(progression_input_ids) == 0:
            progression_input_ids = [self.tokenizer.pad_token_id]
        prior_entity = self.data[index]["temporal_entity_ids"]
        prior_entity_ids = []
        for e in prior_entity:
            if e in self.entity2id:
                prior_entity_ids.append(self.entity2id[e])

        if self.split == "train":
            progressions_ = self.data[index]["temporal_predicates"]
        else:
            progressions_ = []
            if "pro_hyp" in self.op_data[idx]:
                progressions_ = self.op_data[idx]["pro_hyp"]

        # construct progression graph
        graph_info = self.graph_construction(
            prior_observations=prior_observations,
            current_observations=current_observations,
            prior_entity_ids=prior_entity_ids,
            progressions=progressions_,
            labels=labels if self.split == "train" else None,
        )
        gate_labels = [0] * len(labels)
        gather_index = graph_info["gather_index"]
        if self.split == "train":
            for lid, l in enumerate(labels[:-1]):
                if l in graph_info["node_subids"].values():
                    gate_labels[lid] = 1
        item = {
            "image_path": image_path,
            "temporal_image_path": temporal_image_path,
            "input_pixels": pixel_value,
            "input_temporal_pixels": pixel_value_temporal,
            "labels": labels,
            "input_ids": input_ids,
            "progression_input_ids": progression_input_ids,
            "progressions": progressions,
            "split": self.split,
            "observations": self.id2tags[idx],
            "matrix": graph_info["matrix"],
            "node_mask": graph_info["node_mask"],
            "nodes": graph_info["nodes"],
            "gather_index": gather_index,
            "gate_labels": gate_labels,
        }
        if self.split != "train":
            item["report_ids"] = idx
            item["is_temporal"] = len(temporal_image_path) > 0
            item["prior_entity_ids"] = prior_entity_ids
            item["prior_observations"] = prior_observations
        return item

    def __len__(self):
        return len(self.data)

    def graph_construction(
        self,
        prior_observations,
        current_observations,
        prior_entity_ids,
        progressions,
        labels=None,
    ):
        prior_observation_ids = {
            self.entity2id[o] for o in prior_observations if o in self.entity2id
        }
        current_observation_ids = {
            self.entity2id[o] for o in current_observations if o in self.entity2id
        }
        current_relation_ids = {self.relation2id[p] for p in progressions}
        current_relation_ids.add(3)  # S2O
        candidate_entity_ids = set()
        tem_ids = set()

        for (hid, rid), tids in self.triples.items():
            if hid in current_observation_ids and rid in current_relation_ids:
                candidate_entity_ids.update(tids)
                if rid != 3:
                    tem_ids.update(tids)

        nodes = sorted(prior_observation_ids.union(current_observation_ids)) + sorted(
            set(prior_entity_ids).union(candidate_entity_ids)
        )
        node2pos = {node: idx for idx, node in enumerate(nodes)}
        matrix = np.zeros((len(self.id2relation), len(nodes), len(nodes)))

        # prior entity->prior observation
        for eid in prior_entity_ids:
            for oid in prior_observation_ids:
                for rid in self.triples_[(eid, oid)]:
                    matrix[rid, node2pos[oid], node2pos[eid]] = 1

        # prior observation->current observation
        for pid in prior_observation_ids:
            for cid in current_observation_ids:
                for rid in self.triples_[(pid, cid)]:
                    matrix[rid, node2pos[cid], node2pos[pid]] = 1

        # current observation->current observation
        for cid in current_observation_ids:
            for cid2 in current_observation_ids:
                if cid == cid2:
                    continue
                matrix[4, node2pos[cid], node2pos[cid2]] = 1
                matrix[4, node2pos[cid2], node2pos[cid]] = 1

        # current observation->current entity
        for oid in current_observation_ids:
            for eid in candidate_entity_ids:
                for rid in self.triples_[(oid, eid)]:
                    if rid in current_relation_ids:
                        matrix[rid, node2pos[eid], node2pos[oid]] = 1

        node_subids = {
            idx: self.entity2subid[self.id2entity[idx]]
            for idx in candidate_entity_ids
            if self.id2entity[idx] in self.entity2subid
        }
        gather_index = [0] * self.data_args.vocab_size
        for idx, subid in node_subids.items():
            gather_index[subid] = node2pos[idx]

        node_mask = [0] * len(nodes)

        for idx in nodes:
            if idx in candidate_entity_ids:
                node_mask[node2pos[idx]] = 1

            if idx in current_observation_ids:
                node_mask[node2pos[idx]] = -1

        nodes = sorted(nodes, key=lambda x: node2pos[x])

        return {
            "matrix": matrix,
            "nodes": nodes,
            "node_subids": node_subids,
            "gather_index": gather_index,
            "node_mask": node_mask,
            "node2pos": node2pos,
        }
