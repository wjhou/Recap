import json
import os
import argparse

# Run: python src_preprocessing/run_abn_preprocess.py \
    # --mimic_cxr_annotation data/mimic_cxr_annotation.json \
        # --mimic_abn_annotation data/mimic_abn_annotation.json \
            # --image_path data/mimic_cxr_jpg \
                # --output_path data/mimic_abn_annotation_processed.json

parser = argparse.ArgumentParser()
parser.add_argument("--mimic_cxr_annotation", type=str)
parser.add_argument("--mimic_abn_annotation", type=str)
parser.add_argument("--image_path", type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()


ref_anno = json.load(open(args.mimic_cxr_annotation, "r", encoding="utf-8"))
anno = json.load(open(args.mimic_abn_annotation, "r", encoding="utf-8"))

img2report = {}
miss_count = 0
for key in anno:
    for sample in anno[key]:
        for img_path in sample["image_path"]:
            # should be changed according:
            # Here is an example for the Chen et al. (2020) annotation:
            # Chen et al. (2020) Generating Radiology Reports via Memory-driven Transformer. In Proceedings of EMNLP 2020.
            img_path = os.path.join(args.image_path, img_path.replace("files/", ""))
            img_path = img_path.replace(".dcm", ".jpg")
            if not os.path.exists(img_path):
                print("not exist", img_path)
                miss_count += 1
                continue
            image_id = img_path.split("/")[-1].split(".")[0]
            img2report[image_id] = sample["report"]
print(len(img2report), miss_count)

ref_ids = {}
new_anno = {}
for key in ref_anno:
    for sample in ref_anno[key]:
        ref_ids[sample["id"]] = (key, sample)
print(len(ref_ids))

updated_anno = {}
miss_count = 0
for img_id in img2report:
    if img_id not in ref_ids:
        continue
    if ref_ids[img_id][0] not in updated_anno:
        updated_anno[ref_ids[img_id][0]] = []
    new_sample = ref_ids[img_id][1]
    new_sample["report"] = img2report[img_id]
    updated_anno[ref_ids[img_id][0]].append(new_sample)
print([len(val) for val in updated_anno.values()])
with open(args.output_path, "w", encoding="utf-8") as f:
    json.dump(updated_anno, f, indent=4, ensure_ascii=False)
