# Towards Generalizable Vision-Language Robotic Manipulation: A Benchmark and LLM-guided 3D Policy

This repository is the official implementation of [Towards Generalizable Vision-Language Robotic Manipulation: A Benchmark and LLM-guided 3D Policy](https://arxiv.org/abs/2410.01345).

Generalizing language-conditioned robotic policies to new tasks remains a significant challenge, hampered by the lack of suitable simulation benchmarks. In this paper, we address this gap by introducing GemBench, a novel benchmark to assess generalization capabilities of vision-language robotic manipulation policies. As illustrated in the figure below, GemBench incorporates seven general action primitives and four levels of generalization, spanning novel placements, rigid and articulated objects, and complex long-horizon tasks. 

<img src="figures/benchmark_overview.jpg" alt="benchmark" width="800"/>

We evaluate state-of-the-art approaches on GemBench and also introduce a new method. Our approach 3D-LOTUS leverages rich 3D information for action prediction conditioned on language. While 3D-LOTUS excels in both efficiency and performance on seen tasks, it struggles with novel tasks. To address this, we present 3D-LOTUS++ (see figure below), a framework that integrates 3D-LOTUS's motion planning capabilities with the task planning capabilities of LLMs and the object grounding accuracy of VLMs. 3D-LOTUS++ achieves state-of-the-art performance on novel tasks of GemBench, setting a new standard for generalization in robotic manipulation.

<img src="figures/method_overview.jpg" alt="method" width="800"/>

## Installation
See [INSTALL.md](INSTALL.md) for detailed instructions in installation.

## Dataset
The dataset can be found in [Dropbox](https://www.dropbox.com/scl/fo/y0jj42hmrhedofd7dmb53/APlY-eJRqv375beJTIOszFc?rlkey=2txputjiysyg255oewin2m4t2&st=vfoctgi3&dl=0).
Put the dataset in the `data/gembench` folder.
Dataset structure is as follows:
```
- data
    - gembench
        - train_dataset
            - microsteps: 567M, initial configurations for each episode
            - keysteps_bbox: 160G, extracted keysteps data
            - keysteps_bbox_pcd: (used to train 3D-LOTUS)
                - voxel1m: 10G, processed point clouds
                - instr_embeds_clip.npy: instructions encoded by CLIP text encoder
            - motion_keysteps_bbox_pcd: (used to train 3D-LOTUS++ motion planner)
                - voxel1m: 2.8G, processed point clouds
                - action_embeds_clip.npy: action names encoded by CLIP text encoder
        - val_dataset
            - microsteps: 110M, initial configurations for each episode
            - keysteps_bbox_pcd:
                - voxel1m: 941M, processed point clouds
        - test_dataset
            - microsteps: 2.2G, initial configurations for each episode
```

## 3D-LOTUS Policy

### Training
Train the 3D-LOTUS policy end-to-end on the GemBench train split. It takes about 14h with a single A100 GPU.
```bash
sbatch job_scripts/train_3dlotus_policy.sh
```

The trained checkpoints are available [here](https://www.dropbox.com/scl/fo/0g6iz7d7zb524339dgtms/AHS42SO7aPpwut8I8YN8H3w?rlkey=3fwdehsguqsxofzq9kp9fy8fm&st=o25odnuc&dl=0).

### Evaluation
```bash
# both validation and test splits
sbatch job_scripts/eval_3dlotus_policy.sh
```

We use the validation set to select the best checkpoint. The following script summarizes results on the validation split.
```bash
python scripts/summarize_val_results.py data/experiments/gembench/3dlotus/v1/preds/seed100/results.jsonl
```

The following script summarizes results on the test splits of four generalization levels:
```bash
python scripts/summarize_tst_results.py data/experiments/gembench/3dlotus/v1/preds 150000
```


## 3D-LOTUS++ Policy with LLM and VLM

Download llama3-8B model following [instructions here](https://github.com/cshizhe/llama3?tab=readme-ov-file#download), and modify the configuration path in `genrobo3d/configs/rlbench/robot_pipeline.yaml`.

### Training

Train the 3D-LOTUS++ motion planning policy on the GemBench train split. It takes about 14h with a single A100 GPU.
```bash
sbatch job_scripts/train_3dlotusplous_motion_planner.sh
```

The trained checkpoints are available [here]().

### Evaluation

We have three evaluation modes: 
1) groundtruth task planner + groundtruth object grounding + automatic motion planner
2) groundtruth task planner + automatic object grounding + automatic motion planner
3) automatic task planner + automatic object grounding + automatic motion planner

See comments in the following scripts:
```bash
# both validation and test splits
sbatch job_scripts/eval_3dlotus_policy.sh
```