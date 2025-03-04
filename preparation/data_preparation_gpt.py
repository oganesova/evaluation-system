import logging
import pandas as pd
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def prepare_prompts(df):
    df['prompt'] = df['text'].apply(
        lambda x: f"""Please evaluate if the following text is formal or informal. 
        Respond with only one word: 'formal' or 'informal'.
        
        Text: {x}
        
        Response:"""
    )
    return df

def convert_to_dataset(data_frame):
    dataset = Dataset.from_pandas(data_frame)
    logging.info("Dataset created successfully.")
    return dataset

def save_dataset(dataset, output_path):
    try:
        dataset.save_to_disk(output_path)
        logging.info(f"Dataset saved to {output_path}")
        logging.info(f"Dataset size: {len(dataset)}")
    except Exception as e:
        logging.error(f"Error saving dataset: {e}")

if __name__ == "__main__":
    input_file = "dataset/formal_informal_dataset_small.csv"
    output_path = "dataset/dataset_gpt"
    df = load_dataset(input_file)
    df = prepare_prompts(df)
    dataset = convert_to_dataset(df)
    save_dataset(dataset, output_path)
    
    logging.info("Data preparation completed successfully") 