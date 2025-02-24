#!/bin/bash

lm-eval --model hf --model_args pretrained='saved_models/pruned/pruned-llama-7b',parallelize=True,max_memory_per_gpu=10GB                                  --task wikitext --output_path ./benchmarks/results/ppl
lm-eval --model hf --model_args pretrained='saved_models/pruned/pruned-llama2-7b',parallelize=True,max_memory_per_gpu=8GB                                  --task wikitext --output_path ./benchmarks/results/ppl
lm-eval --model hf --model_args pretrained='saved_models/pruned/pruned-llama3.1-8b',parallelize=True,max_memory_per_gpu=4GB                                --task wikitext --output_path ./benchmarks/results/ppl
lm-eval --model hf --model_args pretrained='saved_models/pruned/pruned-llama3.2-1b',parallelize=True,max_memory_per_gpu=10GB                               --task wikitext --output_path ./benchmarks/results/ppl