import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from data_arguments import DataTrainingArguments
from data_process_ende import process_examples
from tokenizer import Tokenizer
from tqdm import tqdm
from PIL import Image
from transformers import GPT2Tokenizer


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
        sample["temporal_report"] = ""
        sample["temporal_predicate"] = []

    for subject_id in tqdm(temporal_ids, desc="Updating Temooral Info"):
        predicate_object = temporal_ids[subject_id]
        predicate = predicate_object["predicate"]

        object_id = predicate_object["object_id"]
        if object_id not in id2sample:
            print(object_id, "Not Found")
        else:
            object_example = id2sample[object_id]
            subject_example = id2sample[subject_id]
            subject_example["temporal_image_path"] = object_example["image_path"]
            subject_example["temporal_report"] = object_example["report"]
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
        id2tags,
        processor,
        observation_category,
        transform=None,
        keep_columns={
            "id",
            "image_path",
            "temporal_image_path",
            "temporal_predicate",
        },
    ) -> None:
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.split = split
        self.dataset = (
            "iu_xray" if "iu_xray" in data_args.annotation_file else "mimic_cxr"
        )
        self.id2tags = id2tags
        examples = {kc: [] for kc in keep_columns}
        samples = annotation[split.replace("valid", "val")]
        temporal_ids = temporal_ids[split.replace("valid", "val")]
        ref_samples = None
        if ref_annotation is not None:
            ref_samples = ref_annotation[split.replace("valid", "val")]
        self.temporal_collection = temporal_ids.keys()
        samples = extract_temporal_info(samples, ref_samples, temporal_ids)
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
            temporal_predicates,
        ) = process_examples(examples=examples)
        self.data = [
            {
                "id": a,
                "image_path": b,
                "temporal_image_path": c,
                "temporal_predicates": d,
            }
            for a, b, c, d in zip(
                idxs,
                image_paths,
                temporal_image_paths,
                temporal_predicates,
            )
        ]
        self.all_index = list(range(len(self.data)))

        self.observation2id = {
            # remove `Support Device` and `No Finding`
            obs: idx
            for idx, obs in enumerate(observation_category[:-2])
        }
        self.transform = transform

    def __getitem__(self, index):
        idx = self.data[index]["id"]

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

        status2id = {"Better": 0, "Worse": 1, "No status change": 2}
        progressions = [0, 0, 0]
        for progression in self.data[index]["temporal_predicates"]:
            progressions[status2id[progression]] = 1

        item = {
            "image_path": image_path,
            "temporal_image_path": temporal_image_path,
            "input_pixels": pixel_value,
            "input_temporal_pixels": pixel_value_temporal,
            "progressions": progressions,
            "split": self.split,
            "observations": self.id2tags[idx],
        }
        if self.split != "train":
            item["report_ids"] = idx
            item["is_temporal"] = len(temporal_image_path) > 0
        return item

    def __len__(self):
        return len(self.data)
