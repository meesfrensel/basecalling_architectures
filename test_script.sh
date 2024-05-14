#!/bin/bash

#SBATCH -J train-model
#SBATCH -o train-own-model-%j.out
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 2-00:00:00
#SBATCH --mem 32G
#SBATCH --gres=gpu:RTX2080Ti:1

echo "Starting test script: neuron selection, only hidden weights (s)"

module load cuda/11.0
echo "Loaded cuda/11.0. Running nvidia-smi:"
nvidia-smi

cd ~/basecalling_architectures
source /shares/bulk/mfrensel/torchclusterenv/bin/activate

echo ""
echo "Starting training"
echo "---"
python ./scripts/train_original.py \
        --data-dir ../datashare/nn_input \
        --output-dir ../datashare/selectionlstm \
        --model own --window-size 2000 --batch-size 64
        #--checkpoint ../datashare/train_output_original_4d/checkpoints/checkpoint_1980000.pt
echo "---"
echo "DONE"

