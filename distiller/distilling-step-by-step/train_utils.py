# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import shutil
import logging

import torch

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments, Trainer
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers.trainer_utils import set_seed

from model_utils import TaskPrefixDataCollatorT5, TaskPrefixTrainerT5, TaskPrefixDataCollatorLlama, TaskPrefixTrainerLlama


def get_config_dir(args):
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}/{args.model_type}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)

    if "t5" in args.from_pretrained:
        print("Model func: T5ForConditionalGeneration")
        model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained, device_map="auto")
    elif "llama" in args.from_pretrained:
        print("Model func: AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(args.from_pretrained, device_map="auto")
    else:
        print("Doesn't recognize model's name. Check model's instance in train_and_evaluate()")

    if args.parallelize:
        model.parallelize()
    
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{run}'  # for training logs

    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    if "t5" in args.from_pretrained:
        training_args = Seq2SeqTrainingArguments(
            output_dir,
            remove_unused_columns = False,
            evaluation_strategy = 'steps',
            eval_steps=args.eval_steps,
            save_strategy='no',
            save_steps=args.eval_steps,
            logging_dir=logging_dir,
            logging_strategy=logging_strategy,
            logging_steps=args.eval_steps,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            gradient_accumulation_steps=args.grad_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            seed=run,
            local_rank=args.local_rank,
            bf16=args.bf16,
            generation_max_length=args.gen_max_len,
            prediction_loss_only=False,
            fp16=True,
        )
    elif "llama" in args.from_pretrained:
        training_args = TrainingArguments(
            output_dir,
            remove_unused_columns=False,
            evaluation_strategy='steps',
            eval_steps=args.eval_steps,
            save_strategy='no',
            save_steps=args.eval_steps,
            logging_dir=logging_dir,
            logging_strategy=logging_strategy,
            logging_steps=args.eval_steps,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            gradient_accumulation_steps=args.grad_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            seed=run,
            local_rank=args.local_rank,
            bf16=args.bf16,
            prediction_loss_only=False,
            fp16=True,  
        )
    else:
        raise ValueError

    if args.model_type == 'task_prefix':
        if "t5" in args.from_pretrained:
            data_collator = TaskPrefixDataCollatorT5(tokenizer=tokenizer, model=model)
        elif "llama" in args.from_pretrained:
            data_collator = TaskPrefixDataCollatorLlama(tokenizer=tokenizer, mlm=False)
        else:
            raise ValueError
    elif args.model_type == 'standard':
        if "t5" in args.from_pretrained:
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        elif "llama" in args.from_pretrained:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        else:
            raise ValueError
    else:
        raise ValueError

    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["test"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }
    

    if args.model_type == 'task_prefix':
        if "t5" in args.from_pretrained:
            trainer = TaskPrefixTrainerT5(**trainer_kwargs)
        elif "llama" in args.from_pretrained:
            trainer = TaskPrefixTrainerLlama(**trainer_kwargs)
        else:
            raise ValueError
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        if "t5" in args.from_pretrained:
            trainer = Seq2SeqTrainer(**trainer_kwargs)
        elif "llama" in args.from_pretrained:
            trainer = Trainer(**trainer_kwargs)
        else:
            raise ValueError
    else:
        raise ValueError
    

    trainer.train()