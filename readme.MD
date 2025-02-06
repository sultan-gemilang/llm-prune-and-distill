# LLM PRUNE AND DISTILL
This is a garuda-ace project combining hard-pruning and various KD methods to speed-up inference while maintaining overall LLM performance.

Currently still WIP...


## Acknowledgements
This repository contains work from other people. 
- TinyLLM ([https://github.com/YikunHan42/TinyLLM](https://github.com/YikunHan42/TinyLLM)).
- Torch-Pruning ([https://github.com/VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)).
- lm-eval ([https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness))

Thanks to the original authors for their great contribution🙇.

## Installation
1. Make new python venv
2. install dependencies from requirements.txt

## Pruning
```bash
python3 prune_llm.py \
    --model meta-llama/Llama-3.2-1B \
    --pruning_ratio 0.3 \ #0.3 ~ 50% pruning
```
- `--model`: Llama model to prune
- `--pruning_ratio` : Pruning ratio (In decimal)

## Benchmark
Bechmarks are tested using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness). Currently there are 3 becnmarks to use,
- Wikitext2 Perplexity
- MMLU
- IFEval

Still deciding which one to choose...

How to use:
```bash
lm-eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B,parallelize=True,max_memory_per_gpu=10GB \
    --task mmlu \
    --num_fewshot 5 \
    --log_samples \
    --output_path './results'
```
- `--model` : Just leave it 'hf' for HF models
- `--model_args` : The model arguments
  - `pretrained` : The model name
  - `parallelize` : Using accelerate to split model between GPUs or CPU
  - `max_memory_per_gpu` : The max memory used to load model, leave some memory for the dataset...
- `--task` : List of benchmarks, see task at [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) (Can be multiple)
- `--num_fewshot` : The number of "shots", usually zero-shot benchmarks or 5
- `--log_samples` : Log the output text of the model
- `--output_path` : Output path for the results

## Latency Test
For latency it can run by running below code

```bash
python3 llama_eval_latency.py --model meta-llama/Llama-3.2-1B --seed 0 --token_size 200 --log True
```

* ``--model`` Llama Model used for inference
  * Note: It is best to use sharded model if you want to run it on Edge
* ``--seed`` Seed used forinference
* ``--token_size`` The size of generated token used for inference
* ``--log`` Log the hardware performance to csv, Default=False

For the text prompt used, you can change it by changing the ``prompt`` variable on the python files.

~WIP~