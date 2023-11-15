# <span style="font-variant:small-caps;">RECAP</span>: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning

This repository is the implementation of [*RECAP: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning*](https://arxiv.org/abs/2310.13864). Before running the code, please install the prerequisite libraries, and follow our instructions to replicate the experiments.

## Overview

Automating radiology report generation can significantly alleviate radiologists' workloads. Previous research has primarily focused on realizing highly concise observations while neglecting the precise attributes that determine the severity of diseases (e.g., small pleural effusion). Since incorrect attributes will lead to imprecise radiology reports, strengthening the generation process with precise attribute modeling becomes necessary. Additionally, the temporal information contained in the historical records, which is crucial in evaluating a patient's current condition (e.g., heart size is unchanged), has also been largely disregarded. To address these issues, we propose **<span style="font-variant:small-caps;">Recap</span>**, which generates precise and accurate radiology reports via dynamic disease progression reasoning. Specifically, **<span style="font-variant:small-caps;">Recap</span>** first predicts the observations and progressions (i.e., spatiotemporal information) given two consecutive radiographs. It then combines the historical records, spatiotemporal information, and radiographs for report generation, where a disease progression graph and dynamic progression reasoning mechanism are devised to accurately select the attributes of each observation and progression. Extensive experiments on two publicly available datasets demonstrate the effectiveness of our model.
![Alt text](figure/overview.png?raw=true "Title")

## Requirements

- `torch==1.9.1`
- `transformers==4.24.0`

## Citation

If you use the <span style="font-variant:small-caps;">Recap</span>, please cite our paper:

```bibtex
@misc{hou2023recap,
      title={RECAP: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning},
      author={Wenjun Hou and Yi Cheng and Kaishuai Xu and Wenjie Li and Jiang Liu},
      year={2023},
      eprint={2310.13864},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
