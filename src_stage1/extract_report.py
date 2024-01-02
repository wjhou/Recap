import json
import sys
from tokenizer import Tokenizer
import os
from tqdm import tqdm

input_path = sys.argv[1]
output_path = sys.argv[2]

if not os.path.exists(output_path):
    os.mkdir(output_path)

print("**************************")
print("input_path: ", input_path)
print("output_path: ", output_path)
print("**************************")

clean_fn = Tokenizer.clean_report_mimic_cxr
reports = {}
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)["train"]
    for report in tqdm(data, desc="Extracting reports"):
        reports[report['id'] + ".txt"] = clean_fn(report['report'])

for idx in tqdm(reports, desc="Writing reports"):
    with open(os.path.join(output_path, idx), "w", encoding="utf-8") as f:
        report = reports[idx]
        f.write(report + '\n')

with open(os.path.join(output_path, "filenames.txt"), "w", encoding="utf-8") as f:
    for idx in reports:
        f.write(os.path.join(output_path, idx) + '\n')
