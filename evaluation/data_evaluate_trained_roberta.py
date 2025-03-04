import logging
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from libs.metric_calculator import MetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate():
    model_path = "models/roberta_formal_informal"
    dataset_path = "dataset/tokenized_dataset_for_training/test"
    
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    test_dataset = load_from_disk(dataset_path)
    
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
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
    logging.info("Starting evaluation for trained RoBERTa model")
    labels ,pred, probs = evaluate()
    print_metrics(labels, pred, probs)