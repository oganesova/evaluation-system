import logging

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizerFast

from libs.metric_calculator import MetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

def init_model(model_type):
    return BertForSequenceClassification.from_pretrained(model_type, num_labels=2)

def load_tokenized_dataset(file_path, batch_size):
    try:
        tokenized_dataset = load_from_disk(file_path)
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)
        logging.info(f"Data loaded successfully : {file_path}")
        return dataloader
    except FileNotFoundError as e:
        logging.error(f"An error occurred during data loading: {e}")
        return

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.numpy())
            all_labels.extend(batch["labels"].numpy())
            probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.numpy())

    return all_labels, all_preds, all_probs

def print_metrics(labels, predictions, probabilities):
    logging.info("Calculating and printing metrics.")
    metrics = MetricsCalculator.calculate_all_metrics(labels, predictions, probabilities)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"AUC ROC: {metrics['auc_roc']:.4f}")

if __name__ == "__main__":
    file_path_dataset = "dataset/tokenized_dataset_bert"
    data_loader = load_tokenized_dataset(file_path_dataset,16)
    model_ = init_model(model_name)
    true_labels, preds, probs = evaluate_model(model_,data_loader)
    print_metrics(true_labels,preds,probs)