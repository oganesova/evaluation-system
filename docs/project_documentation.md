# Text Formality Classification Project Documentation

## Project Structure
```
├── dataset/               # Dataset directory
│   ├── formal_informal_dataset.csv   
│   ├── formal_informal_dataset_small.csv
│   ├── tokenized_dataset_bert/        # Tokenized dataset for BERT
│   ├── tokenized_dataset_roberta/     # Tokenized dataset for RoBERTa
│   ├── tokenized_dataset_for_training/# Dataset for model training
│   └── dataset_gpt/                   # Dataset prepared for GPT-2 evaluation
├── data/                 # Data processing scripts
│   ├── data_preparation.py     # Basic text cleaning and splitting
│   └── data_preparation_spacy.py # NLP processing with spaCy
├── docs/                  # Documentation
│   ├── project_documentation.md  # Technical documentation
│   └── report.md         # Project journey and methodology
├── evaluation/           # Model evaluation scripts
│   ├── data_evaluate_bert.py    # BERT model evaluation
│   ├── data_evaluate_roberta.py # RoBERTa model evaluation
│   ├── data_evaluate_trained_roberta.py # Trained RoBERTa model evaluation
│   ├── data_evaluate_gpt.py      # GPT-2 model evaluation
│   └── libs/                    # Shared libraries
│       └── metric_calculator.py  # Shared metrics calculation
├── models/               # Saved models
│   └── roberta_formality_classifier/ # RoBERTa model artifacts
├── training/            # Training scripts
│   ├── train_model_bert.py    # BERT training script
│   └── train_model_roberta.py # RoBERTa training script
└── requirements.txt     # Project dependencies
```

## Models Used

### BERT Model
- Base model: bert-base-uncased
- Pre-trained on large text corpus
- Efficient at handling context
- Good performance on text classification tasks

### RoBERTa Model
- Base model: roberta-base
- Improved training methodology over BERT
- Better performance on many NLP tasks
- More robust training process

### GPT-2 Model
- Base model: openai-community/gpt2
- Used for zero-shot classification
- Generates text responses for formality classification
- Uses EOS token as padding token
- Temperature set to 0.1 for more deterministic outputs

## Dataset Versions

### Full Dataset (formal_informal_dataset.csv)
- Contains 1,400 lines of text
### Same Small Dataset (formal_informal_dataset_small.csv)
- Contains 210 lines of text

## Data Processing Scripts

### data_preparation_bert.py , data_preparation_train_robert.py, data_preparation_gpt.py
- Basic text cleaning and dataset splitting
- Implements:
  - CSV file reading
  - Null value handling
  - Duplicate removal
  - Text column renaming
  - Label encoding (0 for informal, 1 for formal)
  - Dataset for training splitting into train/val/test
  - Tokenization for different models (BERT, RoBERTa, GPT-2)
  - Dataset format conversion for Hugging Face
  - And only for training dataset splitting - test/train/validation

## Libraries Used

### Core Libraries
- **pandas**: Data processing and manipulation
- **transformers**: BERT model handling and tokenization
- **torch**: Deep learning framework
- **sklearn**: Evaluation metrics and dataset splitting
- **datasets**: Hugging Face dataset handling
- **sentencepiece**: Tokenizer for text processing
- **accelerate**: Hugging Face Accelerate for faster training

## Data Processing Pipeline

1. **Data Preparation**:

- Dataset cleaning and basic preprocessing
- Simple text cleaning , tokenization
- Split into train/validation/test sets
- Label encoding (0 - informal, 1 - formal)

2. **Model Training** (only RoBERTa):

   - Model initialization
   - Tokenization
   - Training configuration
   - Model fine-tuning
   - Model saving

3. **Model Evaluation**:
    1. My trained RoBERTa
    2. BERT, GPT-2
   - Separate evaluation scripts for each model
   - Shared metric calculation - class with static methods
   - Metrics computed:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - ROC AUC
     - Confusion Matrix

## Evaluation

1. **Model-specific Evaluation** (`data_evaluate_bert.py`, `data_evaluate_trained_roberta.py`,`data_evaluate_gpt.py`,):
   - Model loading
   - Prediction generation
   - Model-specific processing

2. **Shared Metrics** (`metric_calculator.py`):
   - Centralized metrics calculation
   - Consistent evaluation across models
   - Standardized reporting

## Results Comparison

Both models are evaluated using the same metrics for fair comparison:
- Accuracy scores
- Precision and Recall values
- F1-scores
- ROC AUC curves
- Confusion matrices

## Methods List

### Data Processing Methods

#### data_preparation_bert.py
- `load_dataset(file_path)`: Loads and prepares dataset from CSV file
- `convert_frame_to_dataset_format(df)`: Converts DataFrame to Hugging Face Dataset format
- `tokenize_function(example)`: Applies BERT tokenization 
- `tokenize_dataset(dataset_format)`: Tokenizes dataset and prepares it for BERT
- `save_tokenized_data(tokenized_data)`: Saves tokenized dataset

#### data_preparation_train_roberta.py
- `load_dataset(file_path)`: Loads and prepares dataset for training
- `convert_frame_to_dataset_format(df)`: Converts DataFrame to Hugging Face Dataset format
- `tokenize_function(example)`: Applies RoBERTa tokenization
- `tokenize_dataset(dataset_format)`: Tokenizes dataset for RoBERTa
- `split_and_save_data(dataset, output_path)`: Splits dataset into train/validation/test and saves

#### data_preparation_gpt.py
- `load_dataset(file_path)`: Loads and prepares dataset
- `prepare_prompts(df)`: Creates prompts for GPT-2
- `convert_to_dataset(data_frame)`: Converts DataFrame to Hugging Face Dataset format
- `save_dataset(dataset, output_path)`: Saves prepared dataset

### Training Methods

#### train_model_with_roberta.py
- `train()`: Main function for RoBERTa model training
  - Loads train and validation datasets
  - Initializes model and tokenizer
  - Configures training parameters
  - Trains model and saves results

### Evaluation Methods

#### data_evaluate_bert.py
- `init_model(model_type)`: Initializes BERT model
- `load_tokenized_dataset(file_path, batch_size)`: Loads tokenized dataset
- `evaluate_model(model, dataloader)`: Performs model evaluation
- `print_metrics(labels, predictions, probabilities)`: Prints evaluation metrics

#### data_evaluate_trained_roberta.py
- `evaluate()`: Main function for evaluating trained RoBERTa model
  - Loads trained model and tokenizer
  - Loads test dataset
  - Performs predictions
  - Calculates and prints metrics
- `print_metrics(labels, predictions, probabilities)`: Prints evaluation metrics

#### data_evaluate_gpt.py
- `load_model_and_tokenizer()`: Loads GPT-2 model and tokenizer
- `get_model_prediction(model, tokenizer, prompt)`: Gets model prediction
- `evaluate()`: Performs model evaluation on dataset
- `print_metrics(labels, predictions, probabilities)`: Prints evaluation metrics

#### metric_calculator.py
Static methods in `MetricsCalculator` class:
- `get_accuracy_metric(true_labels, predictions)`: Calculates accuracy
- `get_precision_metric(true_labels, predictions)`: Calculates precision
- `get_recall_metric(true_labels, predictions)`: Calculates recall
- `get_f1_score_metric(true_labels, predictions)`: Calculates F1 score
- `get_confusion_metric(true_labels, predictions)`: Generates confusion matrix
- `get_roc_auc_score_metric(true_labels, probabilities)`: Calculates ROC AUC
- `calculate_all_metrics(...)`: Computes all metrics in one call



