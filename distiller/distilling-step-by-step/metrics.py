# metrics.py
import numpy as np
import re

def compute_text_acc(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))

def compute_equation_acc(preds, labels):
    preds = [eval_equation(pred) for pred in preds]
    labels = [eval_equation(label) for label in labels]
    return np.mean(np.array(preds) == np.array(labels))

def eval_equation(equation):
    try:
        return eval(equation)
    except:
        return np.nan

def parse_llama_output(text):
    """LLaMA-SPECIFIC: Extract label/rationale from generated text"""
    label_match = re.search(r"Label:\s*(.+?)(\n|$)", text)
    rationale_match = re.search(r"Rationale:\s*(.+?)\nLabel:", text, re.DOTALL)
    
    label = label_match.group(1).strip() if label_match else ""
    rationale = rationale_match.group(1).strip() if rationale_match else ""
    
    return rationale, label

def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Extract labels from LLaMA output
        pred_labels = [parse_llama_output(text)[1] for text in decoded_preds]
        
        # Ground truth labels
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        gt_labels = [text.split("Label: ")[-1].strip() for text in label_texts]

        acc = np.mean(np.array(pred_labels) == np.array(gt_labels))
        return {'accuracy': acc}
    
    return compute_metrics

def compute_metrics_equation(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Extract labels from LLaMA output
        pred_labels = [parse_llama_output(text)[1] for text in decoded_preds]
        
        # Ground truth labels
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        gt_labels = [text.split("Label: ")[-1].strip() for text in label_texts]

        # Numerical evaluation
        pred_values = [eval_equation(p) for p in pred_labels]
        gt_values = [eval_equation(l) for l in gt_labels]
        
        acc = np.mean(np.array(pred_values) == np.array(gt_values))
        return {'accuracy': acc}
    
    return compute_metrics