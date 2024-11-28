#!/bin/bash

set -x
set -e

pwd; hostname; date

cd /mnt/efs/fs-mva/projects/Gembench/robot-3dlotus 

#. $HOME/anaconda3/etc/profile.d/conda.sh
#conda activate gembench

export python_bin=$HOME/anaconda3/envs/gembench/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

expr_dir=data/experiments/gembench/3dlotus/v1
ckpt_step=150000

# validation
${python_bin} genrobo3d/evaluation/eval_simple_policy_server.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 4 \
    --taskvar_file assets/taskvars_train.json \
    --seed 100 --num_demos 20 \
    --microstep_data_dir data/gembench/val_dataset/microsteps/seed100

# test
for seed in {200..600..100}
do
for split in train test_l2 test_l3 test_l4
do
${python_bin} genrobo3d/evaluation/eval_simple_policy_server.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 4 \
    --taskvar_file assets/taskvars_${split}.json \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir data/gembench/test_dataset/microsteps/seed${seed}
done
done
