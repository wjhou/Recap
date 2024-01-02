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
parser.add_argument("--chexbert_label", type=str, required=True, help="the output path")
parser.add_argument("--output_dir", type=str, required=True, help="the output path")
parser.add_argument("--min_count", type=int, default=5, help="min_count")
parser.add_argument(
    "--pmi_threshold", type=float, required=True, help="the output path"
)
parser.add_argument(
    "--temporal_id_dir", type=str, required=True, help="the output path"
)
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
    # observations = [
    #     a for a, b in zip(observation_category, id2observation[idx]) if b == 1
    # ]
    observations = [
        tag2obs(x, y)
        for x, y in zip(id2observation[idx], observation_category)
        if x != 2
    ]
    progressions = set(temporal_id[idx]["predicate"])
    # for observation in temporal_id[idx]["predicate"]:
    #     status = temporal_id[idx]["predicate"][observation]
    #     progressions.update(status)
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
            # p_xy_norm += 1
            progression_ngram_norm[pro] += 1
p_y_norm = sum(sem_stat["ALL"].values())
swords = stopwords.words("english")
p_xy = {
    x[0]: x[1] / progression_ngram_norm[x[0][0]]
    for x in progression_ngram_stat.items()
}
p_x = {x[0]: x[1] / p_x_norm for x in progression_stat.items()}
p_y = {x[0]: x[1] / p_y_norm for x in tem_stat.items()}
pmi = {}
k = 1
max_pmi = defaultdict(float)
avg_pmi = defaultdict(list)
min_pmi = {}
for xy in p_xy:
    observation, ent = xy
    if "No Finding" in observation or "Support Device" in observation:
        continue
    try:
        pmi_xy = math.log(p_xy[xy] ** k / (p_x[observation] * p_y[ent]), 2)
        if pmi_xy <= config.pmi_threshold:
            continue
        pmi[xy] = pmi_xy
        max_pmi[observation] = max(max_pmi[observation], pmi_xy)
        avg_pmi[observation].append(pmi_xy)
        if observation not in min_pmi:
            min_pmi[observation] = pmi_xy
        else:
            min_pmi[observation] = min(min_pmi[observation], pmi_xy)
    except Exception as err:
        print("Error", err, xy)

gloabl_pmi, global_count = 0, 0
for k, v in avg_pmi.items():
    gloabl_pmi += sum(v)
    global_count += len(v)
gloabl_pmi = gloabl_pmi / global_count
median_pmi = {k: np.median(v) for k, v in avg_pmi.items()}
avg_pmi = {k: sum(v) / len(v) for k, v in avg_pmi.items()}
for observation in max_pmi:
    print(
        "Temporal Max/Min/Avg/Median PMI",
        observation,
        round(max_pmi[observation], 3),
        round(min_pmi[observation], 3),
        round(avg_pmi[observation], 3),
        round(median_pmi[observation], 3),
    )
# new_pmi = {}
# for xy in pmi:
#     observation, ent = xy
#     # if pmi[xy] >= 1 and pmi[xy] >= 0.75 * max_pmi[observation]:
#     # if pmi[xy] >= gloabl_pmi and pmi[xy] >= 0.5 * max_pmi[observation]:
#     # if pmi[xy] >= median_pmi[observation] and pmi[xy] >= 1:
#         # if pmi[xy] >= avg_pmi[observation]:
#         # if pmi[xy] >= 1:
#     # if pmi[xy] >= 1:
#     if pmi[xy] >= 0.75 * max_pmi[observation]:
#     # if pmi[xy] >= avg_pmi[observation]:
#         new_pmi[xy] = pmi[xy]
# pmi = new_pmi

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

# max_obs_val = defaultdict(float)
# for key, val in obs2sem.items():
#     obs, entity = key.split("@")
#     max_obs_val[obs] = max(max_obs_val[obs], val)

# obs2sem = sorted(obs2sem.items(), key=lambda x: x[1], reverse=True)
# saved_entity = {"Positive":defaultdict(int), "Negative": defaultdict(int)}
# entity_stat = {}
# for key, val in obs2sem.items():
#     obs, entity = key.split("@")
#     obs_wo_status = obs.split(":")[0]
#     if obs_wo_status not in entity_stat:
#         entity_stat[obs_wo_status] = defaultdict(int)
#     entity_stat[obs_wo_status][entity] += 1
#     # status = "Negative" if "Negative" in obs else "Positive"
#     # if saved_entity[status][entity] >= 2:
#     #     continue
for key, val in obs2sem.items():
    obs, entity = key.split("@")
    # obs_wo_status = obs.split(":")[0]
    # if entity_stat[obs_wo_status][entity] >= 2:
    #     continue
    obs_pmi[obs].append((entity, val))
    # saved_entity[status][entity] += 1

for key, val in new_pairs.items():
    pro, entity = key.split("@")
    new_pmi[pro].append((entity, val))
# new_pmi = {k: sorted(v, key=lambda x: x[1], reverse=True) for k, v in new_pmi.items()}
obs_pmi.update(new_pmi)

# for obs in obs_pmi:
#     if "Positive" in obs:
#         pos_entity = {}
#         neg_entity = {}
#         neg_obs = obs.replace("Positive", "Negative")
#         pos_val = {x[0]: x[1] for x in obs_pmi[obs]}
#         neg_val = {x[0]: x[1] for x in obs_pmi[neg_obs]}
#         for entity in pos_val:
#             if entity not in neg_val or pos_val[entity] > neg_val[entity]:
#                 pos_entity[entity] = pos_val[entity]
#             else:
#                 neg_entity[entity] = neg_val[entity]
#         neg_entity.update(
#             {k: v for k, v in neg_val.items() if k not in pos_entity})
#         obs_pmi[obs] = [(k, v) for k, v in pos_entity.items()]
#         obs_pmi[neg_obs] = [(k, v) for k, v in neg_entity.items()]


obs_pmi = {k: sorted(v, key=lambda x: x[1], reverse=True) for k, v in obs_pmi.items()}
obs_pmi = {k: [x[0] for x in v] for k, v in obs_pmi.items()}

with open(
    os.path.join(config.output_dir, config.dataset, "triples.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(obs_pmi, f, ensure_ascii=False, indent=4)
