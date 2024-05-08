#!/bin/bash

#SBATCH -J basecall-orig-3layer
#SBATCH -o basecall-%j.out
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 1-00:00:00
#SBATCH --mem 32G
#SBATCH --gres=gpu:RTX2080Ti:1

echo "Starting script"

module load cuda/11.0
echo "Loaded cuda/11.0. Running nvidia-smi:"
nvidia-smi

# ROOT_DIR="/shares/bulk/_trainingdata/nanopore/train_output_original_4d_part2"

cd ~/basecalling_architectures
source /shares/bulk/mfrensel/torchclusterenv/bin/activate

echo "Bonito original"
python ./scripts/basecall_original.py \
    --fast5-list ../nanopore_benchmark/static/tasks/fixed_global_task_test_reads.txt \
    --checkpoint ../datashare/doublecheckbonito/checkpoints/checkpoint_232501.pt \
    --output-file ../datashare/doublecheckbonito/out.fastq --model bonito --batch-size 256

echo "Bonito three layers"
python ./scripts/basecall_original.py \
    --fast5-list ../nanopore_benchmark/static/tasks/fixed_global_task_test_reads.txt \
    --checkpoint ../datashare/threelayerbonito/checkpoints/checkpoint_232501.pt \
    --output-file ../datashare/threelayerbonito/out.fastq --model own --batch-size 256

# fast5_files="../nanopore_benchmark/static/tasks/fixed_global_task_test_reads.txt"

# #while IFS= read -r file || [[ -n $file  ]]; do
# #
# #filename=$(basename -- "$file")
# #species=$(basename $(dirname $(dirname "$file")))
# ##filename="${filename%.*}" # without .fast5 extension
# #
# #mkdir -p "${ROOT_DIR}/input/${species}"
# #ln -s "${file}" "${ROOT_DIR}/input/${species}/${filename}"
# #
# #done < $fast5_files

# echo ""
# echo "Starting basecalling"
# echo "---"
# find "${ROOT_DIR}/input" -maxdepth 1 -mindepth 1 -type d -print0 |
#     while IFS= read -r -d '' dir; do
#         species=$(basename "${dir}")
#         python ./scripts/basecall_original.py \
#                 --model sacall \
#                 --fast5-dir "${dir}" \
#                 --checkpoint "${ROOT_DIR}/checkpoints/checkpoint_1600000.pt" \
#                 --output-file "${ROOT_DIR}/basecalls/${species}.fastq" \
#                 --batch-size 128
#     done
# echo "---"
# echo "DONE"

