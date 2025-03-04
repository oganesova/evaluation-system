import logging

import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

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
        return

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

def save_tokenized_data(tokenized_data):
    try:
        tokenized_data.save_to_disk("dataset/tokenized_dataset_bert")
        logging.info("Tokenized dataset saved to 'dataset/tokenized_dataset_bert'.")
    except Exception as e:
        logging.error(f"An error occurred during data saving: {e}")
        return



if __name__ == "__main__":
    dataset_file_path_big = ".dataset/formal_informal_dataset.csv"
    dataset_file_path_small = "dataset/formal_informal_dataset_small.csv"
    dataset_frame = load_dataset(dataset_file_path_small)
    dataset_format_ = convert_frame_to_dataset_format(dataset_frame)
    tokenized = tokenize_dataset(dataset_format_)
    save_tokenized_data(tokenized)