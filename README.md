# <span style="font-variant:small-caps;">RECAP</span>: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning

This repository is the implementation of [RECAP: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning](https://arxiv.org/abs/2310.13864). Before running the code, please install the prerequisite libraries, and follow our instructions to replicate the experiments.

## Update

- [2024/01/13] Checkpoints (Stage 1 and Stage 2) for the MIMIC-ABN dataset are available at [Google Drive](https://drive.google.com/drive/folders/1kGVGeqvGG__jUh8Uds-SMypUD3BPGP5U?usp=sharing)
- [2024/01/12] Checkpoints (Stage 1 and Stage 2) for the MIMIC-CXR dataset are available at [Google Drive](https://drive.google.com/drive/folders/1Tdu1d_OaxiGGoPEpajvHaolzD99mz7u4?usp=sharing)

## Overview

Automating radiology report generation can significantly alleviate radiologists' workloads. Previous research has primarily focused on realizing highly concise observations while neglecting the precise attributes that determine the severity of diseases (e.g., small pleural effusion). Since incorrect attributes will lead to imprecise radiology reports, strengthening the generation process with precise attribute modeling becomes necessary. Additionally, the temporal information contained in the historical records, which is crucial in evaluating a patient's current condition (e.g., heart size is unchanged), has also been largely disregarded. To address these issues, we propose **<span style="font-variant:small-caps;">Recap</span>**, which generates precise and accurate radiology reports via dynamic disease progression reasoning. Specifically, **<span style="font-variant:small-caps;">Recap</span>** first predicts the observations and progressions (i.e., spatiotemporal information) given two consecutive radiographs. It then combines the historical records, spatiotemporal information, and radiographs for report generation, where a disease progression graph and dynamic progression reasoning mechanism are devised to accurately select the attributes of each observation and progression. Extensive experiments on two publicly available datasets demonstrate the effectiveness of our model.
![Alt text](figure/overview.png?raw=true "Title")

## Requirements

- `torch==1.9.1`
- `transformers==4.24.0`

## Data Preparation and Preprocessing

Please download the two datasets: [MIMIC-ABN](https://github.com/zzxslp/WCL/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), and put the annotation files into the `data` folder.

- For observation preprocessing, we use [CheXbert](https://arxiv.org/pdf/2004.09167.pdf) to extract relevant observation information. Please follow the [instruction](https://github.com/stanfordmlgroup/CheXbert#prerequisites) to extract the observation tags.
- For progression preprocessing, we adopt [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/) to extract relevant observation information.
- For entity preprocessing, we use [RadGraph](https://physionet.org/content/radgraph/1.0.0/) to extract relevant entities.
- For CE evaluation, please clone CheXbert into the folder and download the checkpoint [chexbert.pth](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) into CheXbert:

```
git clone https://github.com/stanfordmlgroup/CheXbert.git
```

### Step 1: MIMIC-ABN Data-split Recovery

We recover the data-split of MIMIC-ABN according to `study_id` provided by the MIMIC-CXR dataset. We provide an example code as reference. Please run the following code and change the data location accordingly for preprocessig:

```
python src_preprocessing/run_abn_preprocess.py \
      --mimic_cxr_annotation data/mimic_cxr_annotation.json \
      --mimic_abn_annotation data/mimic_abn_annotation.json \
      --image_path data/mimic_cxr/images/ \
      --output_path data/mimic_abn_annotation_processed.json
```

## Trained Model Weights

Trained model weights on two datasets are available at:

- MIMIC-ABN: [Google Drive](https://drive.google.com/drive/folders/1kGVGeqvGG__jUh8Uds-SMypUD3BPGP5U?usp=sharing)
- MIMIC-CXR: [Google Drive](https://drive.google.com/drive/folders/1Tdu1d_OaxiGGoPEpajvHaolzD99mz7u4?usp=sharing)

## Training and Testing Models

Recap is a two-stage framework as shown the figure above. Here are snippets for training and testing Recap.

### Stage 1: Observation and Progression Prediction

```
chmod +x script_stage1/run_mimic_abn.sh
./script_stage1/run_mimic_abn.sh 1
```

### Stage 2: SpatioTemporal-aware Report Generation

```
chmod +x script_stage2/run_mimic_abn.sh
./script_stage2/run_mimic_abn.sh 1
```

## Citation

If you use the <span style="font-variant:small-caps;">Recap</span>, please cite our paper:

```bibtex
@inproceedings{hou-etal-2023-recap,
    title = "{RECAP}: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning",
    author = "Hou, Wenjun and Cheng, Yi and Xu, Kaishuai and Li, Wenjie and Liu, Jiang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.140",
    doi = "10.18653/v1/2023.findings-emnlp.140",
    pages = "2134--2147",
}
```
