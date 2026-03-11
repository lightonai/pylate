#!/bin/bash
#SBATCH --job-name=cached-xtr-sweep
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-3
#SBATCH --time=4:00:00
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# 4 configs: score x loss
# 0: colbert contrastive
# 1: colbert cached
# 2: xtr contrastive
# 3: xtr cached

SCORES=(colbert colbert xtr xtr)
LOSSES=(contrastive cached contrastive cached)
TEMPERATURES=(1.0 1.0 0.05 0.05)

SCORE=${SCORES[$SLURM_ARRAY_TASK_ID]}
LOSS=${LOSSES[$SLURM_ARRAY_TASK_ID]}
TEMPERATURE=${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}

BS=128
MBS=16
MAX_STEPS=5000
EVAL_STEPS=500

python examples/train/test_cached_xtr.py \
    --score $SCORE --loss $LOSS --temperature $TEMPERATURE \
    --batch_size $BS --mini_batch_size $MBS \
    --max_steps $MAX_STEPS --eval_steps $EVAL_STEPS \
    --wandb --logging_steps 10
