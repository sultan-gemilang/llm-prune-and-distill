#!/bin/bash

lm-eval --model hf --model_args pretrained='baffo32/decapoda-research-llama-7B-hf',parallelize=True,max_memory_per_gpu=10GB,use_fast_tokenizer=False --task wikitext --output_path ./benchmarks/results/ppl
lm-eval --model hf --model_args pretrained='meta-llama/Llama-2-7b-hf',parallelize=True,max_memory_per_gpu=8GB --task wikitext --output_path ./benchmarks/results/ppl
lm-eval --model hf --model_args pretrained='meta-llama/Llama-3.1-8B',parallelize=True,max_memory_per_gpu=4GB --task wikitext --output_path ./benchmarks/results/ppl
lm-eval --model hf --model_args pretrained='meta-llama/Llama-3.2-1B',parallelize=True,max_memory_per_gpu=10GB --task wikitext --output_path ./benchmarks/results/ppl