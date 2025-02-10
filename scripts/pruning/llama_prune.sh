#!/bin/bash

#All model using pruned to 50% pruning based on Torch-Pruning Repo (0.3 ~ 51% pruning)

python3 ./pruning/prune_llm.py --model baffo32/decapoda-research-llama-7B-hf --pruning_ratio 0.3 --save_model ./saved_models/pruned/pruned-llama-7b/
python3 ./pruning/prune_llm.py --model meta-llama/Llama-2-7b-hf --pruning_ratio 0.3 --save_model ./saved_models/pruned/pruned-llama2-7b/
python3 ./pruning/prune_llm.py --model meta-llama/Llama-3.1-8B --pruning_ratio 0.3 --save_model ./saved_models/pruned/pruned-llama3.1-8b/
python3 ./pruning/prune_llm.py --model meta-llama/Llama-3.2-1B --pruning_ratio 0.3 --save_model ./saved_models/pruned/pruned-llama3.2-1b/
