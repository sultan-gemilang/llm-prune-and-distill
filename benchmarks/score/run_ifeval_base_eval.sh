#!/bin/bash

model_list=(
    'baffo32/decapoda-research-llama-7B-hf'
    # 'meta-llama/Llama-2-7b-hf'
    # 'meta-llama/Llama-3.1-8B'
    # 'meta-llama/Llama-3.2-1B'
)

results_path="./benchmarks/results/ifeval"

for model_name in "${model_list[@]}"
do
    lm-eval --model hf\
            --model_args pretrained=$model_name,parallelize=True\
            --task ifeval\
            --log_samples \
            --output_path $results_path

done