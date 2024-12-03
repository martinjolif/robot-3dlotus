# How to create a new task and evaluate it  with 3D-LOTUS

## 1. Installation Instruction

1. Setup an Ubuntu 20.04 exploitation  system for your machine

2. Install general python packages
```bash
conda create -n gembench python==3.10

conda activate gembench

# On CLEPS, first run `module load gnu12/12.2.0`

conda install nvidia/label/cuda-12.1.0::cuda
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

export CUDA_HOME=$HOME/anaconda3/envs/gembench
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

pip install -r requirements.txt

# install genrobo3d
pip install -e .
```
You may have to change the version of flash_attn library depending on the GPU you are using (here it's support T4 GPU, see this [commit](https://github.com/martinjolif/robot-3dlotus/commit/7b17c54abc6ae84468865e09a056ec1848151f7a) for more details). 

3. Install RLBench
```bash
mkdir dependencies
cd dependencies
```

Download CoppeliaSim (see instructions [here](https://github.com/stepjam/PyRep?tab=readme-ov-file#install))
```bash
# change the version if necessary
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

Run these lines in the terminal:
```bash
export COPPELIASIM_ROOT=Gembench/robot-3dlotus/dependencies/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Install Pyrep and RLBench
```bash
git clone https://github.com/cshizhe/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install .
cd ..

# My modified version of RLBench to support new tasks in GemBench
git clone https://github.com/martinjolif/RLBench.git
cd RLBench
pip install -r requirements.txt
pip install .
cd ../..
```

4. Install model dependencies

```bash
cd dependencies

# Please ensure to set CUDA_HOME beforehand as specified in the export const of the section 1
git clone https://github.com/cshizhe/chamferdist.git
cd chamferdist
python setup.py install
cd ..

git clone https://github.com/cshizhe/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
python setup.py install
cd ../..
```

5. Running headless

If you have sudo priviledge on the headless machine, you could follow [this instruction](https://github.com/rjgpinel/RLBench?tab=readme-ov-file#running-headless) to run RLBench.

Otherwise, you can use [singularity](https://apptainer.org/docs/user/1.3/index.html) or [docker](https://docs.docker.com/) to run RLBench in headless machines without sudo privilege.
The [XVFB](https://manpages.ubuntu.com/manpages/xenial/man1/xvfb-run.1.html) should be installed in the virtual image in order to have a virtual screen.

A pre-built singularity image can be downloaded [here](https://www.dropbox.com/scl/fi/wnf27yd4pkeywjk2y3wd4/nvcuda_v2.sif?rlkey=7lpni7d9b6dwjj4wehldq8037&st=5steya0b&dl=0).
Here are some simple commands to run singularity:
```bash
export SINGULARITY_IMAGE_PATH=`YOUR PATH TO THE SINGULARITY IMAGE`
export python_bin=$HOME/miniconda3/envs/gembench/bin/python

# interactive mode
singularity shell --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH

# run script
singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH xvfb-run -a ${python_bin} ...
```

6. Adapt the codebase to your environment

To adapt the codebase to your environment, you may need to modify the following:
- replace everywhere $HOME/codes/robot-3dlotus with your path to robot-3dlotus folder
- replace everywhere the sif_image path to your path to the singularity image nvcuda_v2.sif

7. Enventually download the original dataset 

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
8. Downloads the checkpoints 

The trained checkpoints are available [here](https://www.dropbox.com/scl/fo/0g6iz7d7zb524339dgtms/AHS42SO7aPpwut8I8YN8H3w?rlkey=3fwdehsguqsxofzq9kp9fy8fm&st=eqdd6qvf&dl=0). You should put them in the folder data/experiments/gembench/3dlotus/v1


## 2. Steps to create your new task and evaluate it on the 3D-LOTUS model

1. Create or modify an existing task following the tutorial of the official RLBench repository [here](https://github.com/stepjam/RLBench/tree/master/tutorials). Once you have saved your task .py and .ttm file in the right folder you can continue.
2. Go in `dependencies/RLBench/tools` and run the following command in the terminal:
```
python dataset_generator.py  --tasks reach_target_turorial1
```
replace reach_target_tutorial1 with your task name.
3. It should have created a folder `/tmp/rlbench_data/` with the initial configurations for each episode for your task (.pkl files). Move this in `data/gembench/test_dataset/microsteps/seed200` for example.
4. Now go to `dependancies/RLBench` and do again:
```
pip install .
```
5. Go to `assets/taskvars_instructions_new.json` and modify the file by adding your task name and the text instructions like this in my case:
```
"reach_target_tutorial1+3": [
     "reach the green target",
     "reach the green thing"
  ]
```

Then always in `assets`, create a .json file with your task name: for example I created a `taskvars_new_task.json` file like this:
```
[
 "reach_target_tutorial1+3"
]
```
6. Now you have to encode the text instructions of your task. For that use the `genrobo3d/clip_text_encoder.py` file by replacing with your instruction. It should have created a `embeddings.npy` file.
7. Now go to `data/experiments/gembench/3dlotus/v1/logs/` and modify the `training_config.yaml`file. Replace the line 166 in the VAL_DATASET part with the path of your new `embeddings.npy` file.
8. Now you should be able to evaluate your task! For that go back to `robot-3dlotus` and run the following command in the terminal:
```
python genrobo3d/evaluation/eval_simple_policy_server.py --expr_dir /robot-3dlotus/data/experiments/gembench/3dlotus/v1/ --ckpt_step 150000 --taskvar_file assets/taskvars_new_task.json --seed 200 --num_demos 20 --microstep_data_dir data/gembench/test_dataset/microsteps/seed200 --record_video --video_dir videos
```
You can remove the last parameters `--record_video --video_dir videos` to not record the video, it will be faster!