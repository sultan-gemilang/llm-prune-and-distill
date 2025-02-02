#!/bin/bash

model_list=(
    'baffo32/decapoda-research-llama-7B-hf'
    'meta-llama/Llama-2-7b-hf'
    'meta-llama/Llama-3.1-8B'
    'meta-llama/Llama-3.2-1B'
)

results_path="./benchmarks/results/mmlu"

for model_name in "${model_list[@]}"
do
    accelerate launch \
    -m lm_eval --model hf\
            --model_args pretrained=$model_name,parallelize=True,trust_remote_code=True,use_fast_tokenizer=False\
            --task mmlu\
            --num_fewshot 5\
            --log_samples \
            --output_path $results_path \

done