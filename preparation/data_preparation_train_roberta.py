import logging
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizerFast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        data = data.dropna()
        data = data.drop_duplicates()
        formal_df = data[['formal']].copy()
        formal_df['label'] = 1
        formal_df.rename(columns={'formal': 'text'}, inplace=True)
        informal_df = data[['informal']].copy()
        informal_df['label'] = 0
        informal_df.rename(columns={'informal': 'text'}, inplace=True)
        df = pd.concat([formal_df, informal_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logging.info("Dataset loaded and shuffled successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

def convert_frame_to_dataset_format(df):
    logging.info("Converted DataFrame to Hugging Face Dataset format.")
    return Dataset.from_pandas(df)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

def tokenize_dataset(dataset_format):
    tokenized_dataset = dataset_format.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    logging.info("Tokenization complete.")
    return tokenized_dataset

def split_and_save_data(dataset, output_path):
    splits = dataset.train_test_split(test_size=0.2)
    train_data = splits['train'].train_test_split(test_size=0.125)
    
    final_dataset = {
        "train": train_data['train'],
        "validation": train_data['test'],
        "test": splits['test']
    }

    for split_name, split_data in final_dataset.items():
        split_data.save_to_disk(f"{output_path}/{split_name}")
    
    print(f"Train size: {len(final_dataset['train'])}")
    print(f"Val size: {len(final_dataset['validation'])}")
    print(f"Test size: {len(final_dataset['test'])}")
    return final_dataset

if __name__ == "__main__":
    input_file = "../dataset/formal_informal_dataset_small.csv"
    output_path = "../dataset/tokenized_dataset_for_training"
    
    dataset_frame = load_dataset(input_file)
    dataset_format = convert_frame_to_dataset_format(dataset_frame)
    tokenized = tokenize_dataset(dataset_format)
    split_and_save_data(tokenized, output_path)
    
    logging.info("Data preparation completed successfully")