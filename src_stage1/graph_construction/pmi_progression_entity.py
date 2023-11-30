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
parser.add_argument("--min_count", type=int, default=5, help="min_count")
parser.add_argument("--pmi_threshold", type=float, required=True)
parser.add_argument("--temporal_id_dir", type=str, required=True)
config = parser.parse_args()

print("dataset: ", config.dataset)
clean_fn = Tokenizer.clean_report_mimic_cxr
print(clean_fn)
id2observation, observation_category, _ = Tokenizer.load_tag2ids(
    config.chexbert_label, need_header=True
)
tem_stat = json.load(
    open(
        os.path.join(config.output_dir, config.dataset, "tem_stat.json"),
        "r",
        encoding="utf-8",
    )
)
sem_stat = json.load(
    open(
        os.path.join(config.output_dir, config.dataset, "sem_stat.json"),
        "r",
        encoding="utf-8",
    )
)
with open(
    os.path.join(config.output_dir, config.dataset, "id2entity.json"),
    "r",
    encoding="utf-8",
) as f:
    id2entity = json.load(f)["train"]

with open(config.temporal_id_dir, "r", encoding="utf-8") as f:
    temporal_id = json.load(f)["train"]


def tag2obs(x, y):
    status = "Negative"
    if x == 1:
        status = "Positive"
    return y + ":" + status


id2progression = defaultdict(list)
for idx in temporal_id:
    if len(temporal_id[idx]["predicate"]) == 0:
        continue
    observations = [
        tag2obs(x, y)
        for x, y in zip(id2observation[idx], observation_category)
        if x != 2
    ]
    progressions = set(temporal_id[idx]["predicate"])
    observation2progression = []
    for observation in observations:
        for progression in progressions:
            observation2progression.append(observation + "_" + progression)
    id2progression[idx] = observation2progression

progression_stat = defaultdict(int)
progression_ngram_stat = defaultdict(int)
progression_ngram_norm = defaultdict(int)
p_x_norm = 0
p_xy_norm = 0
swords = stopwords.words("english")
for idx in tqdm(id2progression, desc="Counting Progression"):
    if idx not in id2entity or idx not in id2progression:
        continue
    progressions = id2progression[idx]
    p_x_norm += 1
    for pro in progressions:
        progression_stat[pro] += 1
    entity = id2entity[idx]
    for ent in entity:
        if ent not in tem_stat:
            continue
        for pro in progressions:
            progression_ngram_stat[(pro, ent)] += 1
            progression_ngram_norm[pro] += 1
p_y_norm = sum(sem_stat["ALL"].values())
swords = stopwords.words("english")
p_xy = {
    x[0]: x[1] / progression_ngram_norm[x[0][0]] for x in progression_ngram_stat.items()
}
p_x = {x[0]: x[1] / p_x_norm for x in progression_stat.items()}
p_y = {x[0]: x[1] / p_y_norm for x in tem_stat.items()}
pmi = {}
k = 1
for xy in p_xy:
    observation, ent = xy
    if "No Finding" in observation or "Support Device" in observation:
        continue
    try:
        pmi_xy = math.log(p_xy[xy] ** k / (p_x[observation] * p_y[ent]), 2)
        if pmi_xy <= config.pmi_threshold:
            continue
        pmi[xy] = pmi_xy
    except Exception as err:
        print("Error", err, xy)

new_pairs = {}
for key in pmi:
    new_pairs["@".join(key)] = pmi[key]
with open(
    os.path.join(config.output_dir, config.dataset, "pro2tem.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(new_pairs, f, ensure_ascii=False, indent=4)


with open(
    os.path.join(config.output_dir, config.dataset, "obs2sem.json"),
    "r",
    encoding="utf-8",
) as f:
    obs2sem = json.load(f)

obs_pmi = defaultdict(list)
new_pmi = defaultdict(list)

for key, val in obs2sem.items():
    obs, entity = key.split("@")
    obs_pmi[obs].append((entity, val))

for key, val in new_pairs.items():
    pro, entity = key.split("@")
    new_pmi[pro].append((entity, val))
obs_pmi.update(new_pmi)

obs_pmi = {k: sorted(v, key=lambda x: x[1], reverse=True) for k, v in obs_pmi.items()}
obs_pmi = {k: [x[0] for x in v] for k, v in obs_pmi.items()}

with open(
    os.path.join(config.output_dir, config.dataset, "triples.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(obs_pmi, f, ensure_ascii=False, indent=4)
