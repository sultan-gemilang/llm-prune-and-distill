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


import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer


class TaskPrefixDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.rat_token_id = tokenizer.convert_tokens_to_ids("<RAT>")
        self.lab_token_id = tokenizer.convert_tokens_to_ids("<LAB>")

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        batch["rationale_token_id"] = self.rat_token_id
        batch["label_token_id"] = self.lab_token_id
        return batch

class TaskPrefixTrainer(Trainer):
    def __init__(self, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get special token IDs from collator
        rat_id = inputs["rationale_token_id"]
        lab_id = inputs["label_token_id"]
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Causal LM shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        # Create masks
        rationale_mask = (shift_labels == rat_id)
        label_mask = (shift_labels == lab_id)
        
        # Compute losses
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        rationale_loss = losses[rationale_mask.view(-1)].mean()
        label_loss = losses[label_mask.view(-1)].mean()
        
        total_loss = self.alpha * label_loss + (1 - self.alpha) * rationale_loss
        return (total_loss, outputs) if return_outputs else total_loss  # Fixed 'outputs'