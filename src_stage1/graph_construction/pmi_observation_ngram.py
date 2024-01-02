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
parser.add_argument(
    "--pmi_threshold", type=float, required=True, help="the output path"
)
parser.add_argument("--min_count", type=int, default=5, help="min_count")
parser.add_argument("--max_count", type=int, default=1000, help="max_count")

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
        # if k in sem_stat_keep["ALL"]
        # if v / sum(sem_stat[obs].values()) > 5e-4
    }

# p_x_y = {}
# for obs in sem_stat:
#     p_x_y[obs] = {
#         k: v / sem_stat_all[k]
#         for k, v in sem_stat[obs].items()
#         if k in sem_stat_keep[obs]
#     }

p_y = {k: v / p_y_norm for k, v in sem_stat_all.items()}

pmi = {}
k = 1
rich_entity = defaultdict(list)
max_pmi = defaultdict(float)
min_pmi = {}
avg_pmi = defaultdict(list)
for observation in p_y_x:
    for ent in p_y_x[observation]:
        if ent not in p_y:
            continue
        pmi_xy = math.log(p_y_x[observation][ent] / p_y[ent], 2)
        if pmi_xy <= config.pmi_threshold:
            continue
        pmi[(observation, ent)] = pmi_xy
        max_pmi[observation] = max(max_pmi[observation], pmi_xy)
        if observation not in min_pmi:
            min_pmi[observation] = pmi_xy
        else:
            min_pmi[observation] = min(min_pmi[observation], pmi_xy)
        avg_pmi[observation].append(pmi_xy)
# p_y_x_keep = defaultdict(list)
# for observation in p_y_x:
#     proba = sorted(p_y_x[observation].items(), key=lambda x: x[1], reverse=True)
#     acc_proba = 0
#     for ent, p in proba:
#         if acc_proba < 0.2:
#             acc_proba += p
#             print(observation, ent, p)
#         else:
#             p_y_x_keep[observation].append(ent)

# rich_entity = {k for k, v in rich_entity.items() if len(set([z for z in v if "No Finding" not in z])) > 2}

# except Exception as err:
#     print("Error", err, xy)
# print(p_y_x)
gloabl_pmi, global_count = 0, 0
for k, v in avg_pmi.items():
    gloabl_pmi += sum(v)
    global_count += len(v)
gloabl_pmi = gloabl_pmi / global_count
median_pmi = {k: np.median(v) for k, v in avg_pmi.items()}
avg_pmi = {k: sum(v) / len(v) for k, v in avg_pmi.items()}
for observation in max_pmi:
    print(
        "Spatial Max/Min/Avg/Median PMI",
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
#     # if pmi[xy] >= 1:
#     # if pmi[xy] >= 0.75 * max_pmi[observation]:
#     # if pmi[xy] >= avg_pmi[observation]:
#     if pmi[xy] >= 0.5:
#         # and ent in p_y_x_keep[observation]:
#         # if pmi[xy] >= 0.75 * max_pmi[observation]:
#         # if pmi[xy] >= median_pmi[observation]:
#         new_pmi[xy] = pmi[xy]
# pmi = new_pmi

# sorted_pmi = sorted(pmi.items(), key=lambda x: sem_stat[x[0][0]][x[0][1]], reverse=True)
# sorted_pmi = sorted(pmi.items(), key=lambda x: x[1], reverse=True)
# pmi = {}
# tmp = defaultdict(list)
# saved_entity = {"Positive": set(), "Negative": set()}
# for key, value in sorted_pmi:
#     observation, ent = key
#     # if "No Finding" in observation:
#     #     continue
#     if ent not in saved_entity[observation.split(":")[1]]:
#         tmp[observation].append((ent, value))
#         saved_entity[observation.split(":")[1]].add(ent)

# # for key, value in sorted_pmi:
# #     observation, ent = key
# #     if "No Finding" not in observation:
# #         continue
# #     query = observation.split(":")[1]
# #     if query == "Positive":
# #         query = "Negative"
# #     else:
# #         query = "Positive"
# #     if ent not in saved_entity[query]:
# #         tmp[observation].append((ent, value))
# #         saved_entity[query].add(ent)


# for key in tmp:
#     for ent, value in tmp[key]:
#         pmi[(key, ent)] = value

new_pairs = {}
for key in pmi:
    # if key[1] in rich_entity:
    #     continue
    new_pairs["@".join(key)] = pmi[key]

with open(
    os.path.join(config.output_dir, config.dataset, "obs2sem.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(new_pairs, f, ensure_ascii=False, indent=4)
