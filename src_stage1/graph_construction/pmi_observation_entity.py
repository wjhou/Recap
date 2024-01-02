from tqdm import tqdm
import pandas as pd
import argparse
import json
import math
from collections import defaultdict
import os
from nltk.corpus import stopwords
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="the name of dataset")
parser.add_argument("--chexbert_label", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--pmi_threshold", type=float, required=True)
parser.add_argument("--min_count", type=int, default=5, help="min_count")

config = parser.parse_args()

print("dataset: ", config.dataset)
clean_fn = Tokenizer.clean_report_mimic_cxr
print(clean_fn)
id2observation, observation_category, _ = Tokenizer.load_tag2ids(
    config.chexbert_label, need_header=True
)
print(len(id2observation), observation_category)

sem_stat = json.load(
    open(
        os.path.join(config.output_dir, config.dataset, "sem_stat.json"),
        "r",
        encoding="utf-8",
    )
)
tem_stat = json.load(
    open(
        os.path.join(config.output_dir, config.dataset, "tem_stat.json"),
        "r",
        encoding="utf-8",
    )
)

swords = stopwords.words("english")
sem_stat_keep = {
    k: {
        subk
        for subk, subv in v.items()
        if subk not in swords
        and not subk.startswith("_")
        and subv >= config.min_count
        and subk not in tem_stat
    }
    for k, v in sem_stat.items()
}


with open(
    os.path.join(config.output_dir, config.dataset, "id2entity.json"),
    "r",
    encoding="utf-8",
) as f:
    id2entity = json.load(f)
observation_stat = defaultdict(int)
observation_ngram_stat = defaultdict(int)
observation_ngram_norm = defaultdict(int)

sem_stat_all = sem_stat["ALL"]
p_y_norm = sum(sem_stat_all.values())
sem_stat.pop("ALL")
p_y_x = {}
for obs in sem_stat:
    p_y_x[obs] = {
        k: v / sum(sem_stat[obs].values())
        for k, v in sem_stat[obs].items()
        if k in sem_stat_keep[obs]
    }

p_y = {k: v / p_y_norm for k, v in sem_stat_all.items()}

pmi = {}
k = 1
for observation in p_y_x:
    for ent in p_y_x[observation]:
        if ent not in p_y:
            continue
        pmi_xy = math.log(p_y_x[observation][ent] / p_y[ent], 2)
        if pmi_xy <= config.pmi_threshold:
            continue
        pmi[(observation, ent)] = pmi_xy

new_pairs = {}
for key in pmi:
    new_pairs["@".join(key)] = pmi[key]

with open(
    os.path.join(config.output_dir, config.dataset, "obs2sem.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(new_pairs, f, ensure_ascii=False, indent=4)
