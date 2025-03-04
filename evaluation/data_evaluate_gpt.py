import logging
from datasets import load_from_disk
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from libs.metric_calculator import MetricsCalculator


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer():
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def get_llm_judgment(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
        num_return_sequences=1,
        temperature=0.1,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip().lower()

    logits = outputs.scores[-1]
    probs = softmax(logits, dim=-1)
    predicted_label = 1 if "formal" in response else (0 if "informal" in response else -1)
    probability = probs.max().item() if predicted_label != -1 else 0.5

    return predicted_label, probability

def evaluate_with_llm():
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_from_disk("dataset/dataset_gpt")

    logging.info(f"Loaded dataset with {len(dataset)} examples.")

    all_preds = []
    all_labels = []
    all_probs = []

    for example in tqdm(dataset, desc="Evaluating with LLM as Judge"):
        prompt = example["prompt"]
        true_label = example["label"]

        prediction, probability = get_llm_judgment(model, tokenizer, prompt)

        if prediction != -1:
            all_preds.append(prediction)
            all_labels.append(true_label)
            all_probs.append(probability)

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
    logging.info("Evaluation with LLM-as-a-judge")
    labels, preds, probs = evaluate_with_llm()
    print_metrics(labels, preds, probs)
