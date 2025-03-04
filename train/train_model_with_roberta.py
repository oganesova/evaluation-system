import logging
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train():

    train_dataset = load_from_disk("../dataset/tokenized_dataset_for_training/train")
    val_dataset = load_from_disk("../dataset/tokenized_dataset_for_training/validation")

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    training_args = TrainingArguments(
        output_dir="../models/roberta_formal_informal",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        save_strategy="epoch",
        logging_dir="../logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    try:
        logging.info("Starting model training")
        trainer.train()
        model.save_pretrained("../models/roberta_formal_informal")
        tokenizer.save_pretrained("../models/roberta_formal_informal")
        logging.info("Model training complete and saved!")
    except Exception as e:
        logging.error(f"Error during training: {e}")

if __name__ == "__main__":
    logging.info("Starting training")
    train() 