# Formality Detection Project Report

Data Prep for BERT
___
![image](https://github.com/user-attachments/assets/863852f1-3799-415e-9670-ee5392cabf36)
___
Data Prep for RoBERTa
___
![image](https://github.com/user-attachments/assets/967a3a75-a39a-464c-ad3d-580247a421e2)
___
Data Prep for RoBERTa to train
___
![image](https://github.com/user-attachments/assets/967a3a75-a39a-464c-ad3d-580247a421e2)
___
Data Prep for GPT-2
___
![image](https://github.com/user-attachments/assets/967a3a75-a39a-464c-ad3d-580247a421e2)
___

Training - RoBERTA
___
![image](https://github.com/user-attachments/assets/af824865-ae85-4578-8a2e-e5c33281916e)
![image](https://github.com/user-attachments/assets/f1b967ea-0726-493d-b455-1b94169684fc)

___
Evaluating - BERT
___
![image](https://github.com/user-attachments/assets/ed23422e-3e39-44fd-bfba-54899b84b5a4)
___
Evaluating - RoBERTa
___
___
Evaluating - RoBERTa with training
___
___
Evaluating - GPT-2
___
___
## My Project Journey and Implementation Steps
___

### Dataset Selection and Preparation
___
In my first step in my project, I started with the selection of a proper dataset. I searched a bit and understood that we have a lot of public datasets available, and even we can create synthetic data, which prevents excessive manual data preparation.
Then, I found a very good dataset in Kaggle, which has both formal and informal documents. 
I downloaded it and proceeded to clean the data.

kaggle-dataset
I downloaded it from here -- > https://www.kaggle.com/datasets/shiromtcha/formal-and-informal-phrases

![image](https://github.com/user-attachments/assets/5d93cf99-dea2-458e-813f-60e2809748a7)


After I cleaned the dataset, remove duplicates, none values (for all models), also 
I split dataset for training into three sets:
- Training set
- Test set
- Validation set

### Model Selection
___
Initially, I wanted to use **LLaMA** (Large Language Model Meta AI)but I discovered that LLaMA requires a specific license from Meta, and the licensing process would take time for requesting and approval.

Next, I looked into various models and chose to train on **BERT**. I chose BERT because:
- It is pre-trained on vast amounts of text
- Has proven effectiveness on various natural language processing tasks
- Particularly strong in text classification
- Efficient at handling context
- Provides high accuracy
- Most importantly, it's openly available without licensing restrictions

For comparison, I also selected **RoBERTa** as my second model. RoBERTa was chosen because:
- It's an optimized version of BERT with improved training methodology
- Removes the Next Sentence Prediction (NSP) objective
- Uses dynamic masking patterns
- Trains with larger batches
- Uses a larger vocabulary
- Shows consistently better performance on many NLP tasks

And for last comparison i selected **GPT-2** because:
- It's a powerful language model that can be used for zero-shot classification
- Provides more natural and context-aware responses
- Can handle complex language patterns without fine-tuning
- Offers a different approach to classification through text generation
- Helps understand how well a general-purpose language model performs on this specific task

### Performance Metrics
___
At the same time, I started searching for the metrics to be used in order to assess the performance of the model. I chose the following metrics:

- **Accuracy**: It assesses how often the model classifies the text correctly.
- **Recall**: Assesses the model's ability to recognize all positive examples, i.e., correctly classify formal and informal texts.
- **Precision**: Defines the correctness of the model in marking positive examples as "formal" or "informal."
- **F1-score**: Harmonic mean of precision and recall, with an aim to achieve a balance of both measures.
- **Confusion Matrix**: Illustrates how many times the model gets things right or wrong, giving even more insight into its performance.
- **AUC-ROC**: The area under the curve, which assists in measuring the performance of the model at all possible thresholds.
This is important for measuring the trade-off between precision and recall.

These metrics are important since they give us a complete picture of the performance of the model and consider many aspects of its behavior. Having all these metrics enables us to have a better idea of how well the model performs our goals and tasks.

### Model Comparison
___
After completing the initial training with BERT, I implemented RoBERTa for comparison. RoBERTa showed several interesting characteristics:
- More robust training process due to its improved methodology
- Generally faster convergence during training
- Better handling of complex language patterns
- Improved performance on formal language detection

I repeated all the evaluation steps for both models, ensuring a fair comparison using identical metrics and test data. The results showed that while both models performed well, RoBERTa demonstrated some advantages in:
- Accuracy on formal text classification
- Handling of complex sentence structures
- Overall robustness of predictions

All the metrics were calculated and reported for both models, allowing for a comprehensive comparison of their strengths and weaknesses.

## Conclusion
___
This has been an interesting and educational project. In three days, I learned many algorithms and methods,
discovered helpful video lectures, and learned so much more about machine learning. I will continue learning and expanding my knowledge in this field.
This has been a rich experience.

Thank you for reading! It was an exciting ride into the world of text processing and machine learning. 

## P.S
___

In my screenshots, you can see small numbers like 180,
but my actual dataset contains 1,400 lines.
I created a smaller version of my dataset for testing purposes
because, unfortunately, my computer couldn't handle the full
dataset—it was showing an estimated runtime of 10 hours or more.
To ensure everything works correctly before sharing it for review,
I shortened the dataset **(formal_informal_dataset_small.csv).**
However, the code is fully optimized and works just as
effectively on the full dataset **(formal_informal_dataset.csv).**
