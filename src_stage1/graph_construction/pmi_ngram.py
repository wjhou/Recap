from tqdm import tqdm
from collections import defaultdict
import json
import argparse
import os
from constants import TEM_KEYWORDS
from nltk.corpus import stopwords, wordnet
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="the name of dataset")
parser.add_argument("--output_dir", type=str, required=True, help="the output path")
parser.add_argument("--min_count", type=int, default=5, help="min_count")
parser.add_argument("--chexbert_label", type=str, required=True, help="the output path")


def tag2obs(x, y):
    status = "Negative"
    if x == 1:
        status = "Positive"
    return y + ":" + status


def filter_fn(s):
    punc = set(".,?;*!%^&_+():-\[\]\{\}")
    numbers = set("0123456789")
    chars = set(s)
    if (
        len(s) <= 1
        or len(chars.intersection(punc)) > 0
        or len(chars.intersection(numbers)) > 0
    ):
        return False
    return True


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from tokenizer import Tokenizer

    config = parser.parse_args()
    dataset = config.dataset
    min_count = config.min_count
    min_frequent = 3 if "mimic_abn" in dataset else 10

    print("dataset: ", dataset)
    clean_fn = Tokenizer.clean_report_mimic_cxr

    id2observation, observation_category, _ = Tokenizer.load_tag2ids(
        config.chexbert_label, need_header=True
    )
    with open("../%s/annotation.json" % config.dataset, "r", encoding="utf-8") as f:
        f_read = json.load(f)
    with open(
        "./data/%s_sentence_level_observation.json" % config.dataset,
        "r",
        encoding="utf-8",
    ) as f:
        sent_f_read = json.load(f)["train"]

    collect_ids = set()
    for sample in f_read["train"]:
        idx = "/".join(sample["image_path"][0].split("/")[:-1]) + ".txt"
        collect_ids.add(idx)
    data = []
    sem_path = os.path.join(config.output_dir, config.dataset, "sem.txt")
    spatial_keywords = set()
    key_tuple = defaultdict(int)
    window = 0
    if not os.path.exists(sem_path):
        with open(
            "/home/wenjun/repo/report_gen/physionet.org/files/radgraph/1.0.0/MIMIC-CXR_graphs.json",
            "r",
            encoding="utf-8",
        ) as f:
            radgraph = json.load(f)
            for key in radgraph:
                if key in collect_ids:
                    for entity_idx in radgraph[key]["entities"]:
                        entity = radgraph[key]["entities"][entity_idx]
                        if len(entity["relations"]) <= 0:
                            continue
                        for relation in entity["relations"]:
                            if "modify" in relation or "located_at" in relation:
                                # if "modify" in relation:
                                k1 = [
                                    z.strip().lower()
                                    for z in entity["tokens"].split()
                                    if filter_fn(z.strip())
                                ]
                                k2 = [
                                    z.strip().lower()
                                    for z in radgraph[key]["entities"][relation[1]][
                                        "tokens"
                                    ].split()
                                    if filter_fn(z.strip())
                                ]
                                if len(k1) == 1 and len(k2) == 1:
                                    ix1 = entity["start_ix"]
                                    ix2 = radgraph[key]["entities"][relation[1]][
                                        "start_ix"
                                    ]
                                    window = max(window, abs(ix1 - ix2))
                                    key_tuple[(k1[0], k2[0])] += 1

        with open(sem_path, "w", encoding="utf-8") as f:
            f.write(str(window) + "\n")
            for a in spatial_keywords:
                f.write(a + "\n")

            for b in key_tuple:
                f.write("-modify-".join(b) + "," + str(key_tuple[b]) + "\n")
    else:
        with open(sem_path, "r", encoding="utf-8") as f:
            window = int(f.readline().strip())
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                if "-modify-" in line:
                    line = line.split(",")
                    key_tuple[tuple(line[0].split("-modify-"))] = int(line[1])
                else:
                    spatial_keywords.add(line)
    spatial_keywords = set()

    for k1, k2 in key_tuple:
        if key_tuple[(k1, k2)] <= (50 if "mimic_abn" in dataset else 50):
            # if key_tuple[(k1, k2)] <= 50:
            continue
        # if k1 not in TEM_KEYWORDS and key_tuple[(k1, k2)] >= (50 if "mimic_abn" in dataset else 200):
        # if key_tuple[(k1, k2)] >= (50 if "mimic_abn" in dataset else 200):
        #     spatial_keywords.add(k1)
        if k1 not in TEM_KEYWORDS:
            spatial_keywords.add(k1)
        if k2 not in TEM_KEYWORDS:
            spatial_keywords.add(k2)

    window = min(3, window)
    head2tail = defaultdict(set)
    for k1, k2 in key_tuple:
        head2tail[k1].add(k2)

    id2report = {}
    split2id2report = {"val": {}, "test": {}}
    for sample in tqdm(f_read["train"], desc="Loding Reports"):
        report = clean_fn(sample["report"])
        id2report[sample["id"]] = report.split()
    for sample in tqdm(f_read["val"], desc="Loding Reports"):
        report = clean_fn(sample["report"])
        split2id2report["val"][sample["id"]] = report.split()
    for sample in tqdm(f_read["test"], desc="Loding Reports"):
        report = clean_fn(sample["report"])
        split2id2report["test"][sample["id"]] = report.split()
    split2id2report["train"] = id2report

    sem_stat_all = {
        "No Finding:Positive": defaultdict(int),
        "No Finding:Negative": defaultdict(int),
    }
    sem_stat = defaultdict(int)
    tem_stat = defaultdict(int)
    id2entity = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }
    id2entity_all = {}
    sword = {"."}
    sword.update(stopwords.words("english"))
    for idx in id2report:
        observations = id2observation[idx]
        observations = [
            tag2obs(x, y) for x, y in zip(observations, observation_category) if x != 2
        ]
        tokens = id2report[idx]
        for token in tokens:
            if token in sword:
                continue
            sem_stat[token] += 1
            if token in TEM_KEYWORDS:
                tem_stat[token] += 1
        if idx not in sent_f_read:
            continue
        sentences = sent_f_read[idx]
        no_finding = (
            "No Finding:Positive"
            if "No Finding:Positive" in observations
            else "No Finding:Negative"
        )
        for pos in sentences:
            tokens = [
                tok for tok in sentences[pos]["sentence"].split() if tok not in sword
            ]
            observation = [
                o for o in sentences[pos]["observation"] if o in observations
            ]
            # if len(observation) == 0:
            if True:
                observation.append(no_finding)
            for obs in observations:
                if obs not in sem_stat_all:
                    sem_stat_all[obs] = defaultdict(int)
                for i, token in enumerate(tokens):
                    if obs in observation and token in spatial_keywords:
                        sem_stat_all[obs][token] += 1
                    else:
                        sem_stat_all[obs]["_" + token] += 1

    # max_stat = defaultdict(int)
    # for obs in sem_stat_all:
    #     # if "No Finding" in obs:
    #     #     continue
    #     flip_obs = (
    #         obs.replace("Positive", "Negative")
    #         if "Positive" in obs
    #         else obs.replace("Negative", "Positive")
    #     )
    #     stat = {}
    #     for k, v in sem_stat_all[obs].items():
    #         if k.startswith("_"):
    #             continue
    #         count = v + sem_stat_all[flip_obs].get(k, 0)
    #         max_stat[k] = max(max_stat[k], count)
    # new_sem_stat_all = {}
    # for obs in sem_stat_all:
    #     if "No Finding" in obs:
    #         continue
    #     flip_obs = (
    #         obs.replace("Positive", "Negative")
    #         if "Positive" in obs
    #         else obs.replace("Negative", "Positive")
    #     )
    #     stat = {}
    #     for k, v in sem_stat_all[obs].items():
    #         count = v + sem_stat_all[flip_obs].get(k, 0)
    #         # count2 = sem_stat_all["No Finding:Negative"].get(k, 0) + sem_stat_all[
    #         #     "No Finding:Positive"
    #         # ].get(k, 0)
    #         # if not k.startswith("_") and count <= sem_stat[k] // len(observation_category) * 2:
    #         # if not k.startswith("_") and (count <= count2 * 0.5):
    #         if not k.startswith("_") and count <= max_stat[k] * 0.5:
    #             # print(obs, k, count, max_stat[k] * 0.25, count2 * 0.75)
    #             # if not k.startswith("_") and count <= count2:
    #             k = "_" + k
    #             stat[k] = v + sem_stat_all[obs].get(k, 0)
    #         if k not in stat:
    #             stat[k] = v
    #     new_sem_stat_all[obs] = stat
    # new_sem_stat_all["No Finding:Positive"] = sem_stat_all["No Finding:Positive"]
    # new_sem_stat_all["No Finding:Negative"] = sem_stat_all["No Finding:Negative"]
    # sem_stat_all = new_sem_stat_all

    sem_stat_all["ALL"] = sem_stat
    sem_stat_all = {
        k: {
            subk: subv
            for subk, subv in v.items()
            if subv >= min_frequent and subk not in sword
        }
        for k, v in sem_stat_all.items()
    }
    for split in split2id2report:
        for idx in split2id2report[split]:
            tokens = split2id2report[split][idx]
            for token in tokens:
                if (
                    token in TEM_KEYWORDS or token in spatial_keywords
                ) and token not in id2entity[split][idx]:
                    id2entity[split][idx].append(token)

    id2entity_all.update(id2entity)
    with open(
        os.path.join(config.output_dir, config.dataset, "id2entity.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(id2entity_all, f, indent=4, ensure_ascii=False)
    with open(
        os.path.join(config.output_dir, config.dataset, "sem_stat.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(sem_stat_all, f, indent=4, ensure_ascii=False)
    with open(
        os.path.join(config.output_dir, config.dataset, "tem_stat.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(tem_stat, f, indent=4, ensure_ascii=False)
