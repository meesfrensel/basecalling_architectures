#!/bin/bash

#SBATCH -o prune-log-%j.out
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 1-00:00:00
#SBATCH --mem 32G
#SBATCH --gres=gpu:RTX2080Ti:1

echo "Starting prune&tune script"

module load cuda/11.0
echo "Loaded cuda/11.0. Running nvidia-smi:"
nvidia-smi

cd ~/basecalling_architectures
source /shares/bulk/mfrensel/torchclusterenv/bin/activate

echo "Help output:"
python ./scripts/prune_finetune.py --help

echo ""
echo "Starting prune&tune"
echo "---"
python ./scripts/prune_finetune.py \
        --checkpoint ../datashare/pretrained_models/recreation/bonito_2000/inter/checkpoint.pt \
        --data-dir ../datashare/nn_input \
        --num-iterations 1 \
        --output-dir "/shares/bulk/mfrensel/prunetune/prunetune_bonito_${*// /_}" \
        --prune $@
        #--selfish-rnn --density 0.2 --death-rate 0.3 --death SET --redistribution momentum \
        #--output-dir /shares/bulk/mfrensel/selfishrnn
echo "---"
echo "DONE"
