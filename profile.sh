#!/bin/bash

#SBATCH -J profile-basecaller
#SBATCH -o profile-%j.out
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 2:00:00
#SBATCH --mem 64G
#SBATCH --gres=gpu:RTX2080Ti:1

echo "Starting script"

module load cuda/11.0
echo "Loaded cuda/11.0. Running nvidia-smi:"
nvidia-smi

echo "Cleaning up /tmp and making links to /work/tmp"
rm /tmp/nsys-report-*
rm /tmp/injection_config_*
mkdir -p /work/tmp/nvidia
ln -s /work/tmp/nvidia /tmp/nvidia

ROOT_DIR="/shares/bulk/_trainingdata/nanopore/train_output_original_4d_part2"

cd ~/basecalling_architectures
source /shares/bulk/mfrensel/torchclusterenv/bin/activate

echo ""
echo "Starting profiled basecalling"
echo "---"
TMPDIR=/work/tmp nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s none -o nsight_report -f true -x true --gpu-metrics-device=all --gpu-metric-set=tu10x \
        python ./scripts/basecall_original.py \
                --model sacall \
                --fast5-dir "${ROOT_DIR}/../data/wick/Klebsiella_pneumoniae-KSB1_6F/fast5" \
                --checkpoint "${ROOT_DIR}/checkpoints/checkpoint_1600000.pt" \
                --output-file /dev/null \
                --batch-size 256
echo "---"

echo "Removing ~/.nsight-systems"
rm -r ~/.nsight-systems
echo "DONE"

