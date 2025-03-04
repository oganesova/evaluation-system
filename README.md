# Improving Writing Assistance at JetBrains AI
# Text Formality Classifier

![image](https://github.com/user-attachments/assets/27bb86c1-f47b-4a1e-983e-6a0f563667fc)

A project for evaluating text classifying (formal or informal) using BERT,RoBERTa, GPT-2 .

## Models Implemented

BERT, RoBERTa , GPT-2

## Requirements

- Python 3.8+
- pip (Python package manager)

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/oganesova/evaluation-system.git
   cd evaluation-system
   ```

2. Set up the environment and install dependencies:
```bash

python -m venv venv

venv\Scripts\activate |  source venv/bin/activate

pip install -r requirements.txt
```
## Run Data Preparation scripts
3. First run BERT data preparation, after that RoBERTa, and GPT-2.
   It will save everything in dataset dir , separately.

```bash
python data/data_preparation_bert.py
python data/data_preparation_train_roberta.py
python data/data_preparation_gpt.py
```

## Training Model - RoBERTa
4. Please run this scripts to train model.
```bash

python train/train_model_with_roberta.py

```

I try to push saved model on GitHub, but it almost 9GB and more.
RoBERTa model will save in models dir 


## Model Evaluation

5. Evaluate each model's performance:


```bash
python evaluation/data_evaluate_bert.py
python evaluation/data_evaluate_trained_roberta.py
python evaluation/data_evaluate_gpt.py

```

## Dataset Options

The project supports two dataset versions:
- `formal_informal_dataset.csv`: Full dataset (1,400 lines)
- `formal_informal_dataset_small.csv`: Same dataset (220 lines) , but small version.
  (I create this for fast testing , because 1400 lines of data training will take so much longer/ 10 hours and more)


## Development Environment

The project is set up with:
- IntelliJ IDEA configuration (`.idea/` directory)
- Python virtual environment (`venv/` directory)
- IntelliJ IDEA module configuration (`evaluation-system.iml`)

## Documentation

1. `docs/project_documentation.md`: Technical details and implementation
2. `docs/report.md`: Project methodology and research process
3. `requirements.txt`: All dependencies with versions

## Model Result (Only RoBERTa, I Trained it just for interest, to compare)

Model are saved in:
- RoBERTa: `models/roberta_formality_classifier/`

## Performance Metrics

Both models are evaluated using:
- Accuracy
- Precision/Recall
- F1-score
- ROC AUC
- Confusion Matrix

Detailed results can be found in the evaluation outputs (docs/report.md). 
