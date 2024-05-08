#!/bin/bash

set -euo pipefail

cfgs=(
    "0=0.9" "3=0.4" "4=0.4"
    #"4=0.5" "4=0.6" "4=0.7" "4=0.8" "4=0.9"
    #"3=0.5" "3=0.6" "3=0.7" "3=0.8" "3=0.9"
    #"2=0.4" "2=0.5" "2=0.6" "2=0.7" "2=0.8" "2=0.9"
    #"1=0.3" "1=0.4" "1=0.5" "1=0.6" "1=0.7" "1=0.8" "1=0.9"
    #"0=0.3" "0=0.4" "0=0.5" "0=0.6" "0=0.7" "0=0.8"
    #"3=0.5 4=0.5" "3=0.5 4=0.7" "3=0.5 4=0.8" "3=0.5 4=0.9"
    #"3=0.7 4=0.5" "3=0.7 4=0.7" "3=0.7 4=0.8" "3=0.7 4=0.9"
    #"3=0.8 4=0.5" "3=0.8 4=0.7" "3=0.8 4=0.8" "3=0.8 4=0.9"
    #"3=0.9 4=0.5" "3=0.9 4=0.7" "3=0.9 4=0.8" "3=0.9 4=0.9"
    #"0=0.5 1=0.5 2=0.5 3=0.5 4=0.5" "0=0.6 1=0.6 2=0.6 3=0.6 4=0.6"
    #"0=0.7 1=0.7 2=0.7 3=0.7 4=0.7" "0=0.8 1=0.8 2=0.8 3=0.8 4=0.8"
    #"0=0.9 1=0.9 2=0.9 3=0.9 4=0.9"
    #"1=0.5 2=0.5 3=0.5" "1=0.6 2=0.6 3=0.6" "1=0.7 2=0.7 3=0.7" "1=0.8 2=0.8 3=0.8" "1=0.9 2=0.9 3=0.9"
)

for cfg in "${cfgs[@]}"; do
    # Prune & finetune
    sbatch -J "prune_${cfg// /_}" ./prune_script.sh $cfg
    # Basecall
    # Evaluate
    #sbatch -J "eval_${cfg// /_}" ./eval_script.sh
done
