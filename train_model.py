import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import transformers
import torch
from datasets import Dataset
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import os
import json

def train_model():

    # Define path to label mapping
    label_mapping_path = './label_mapping_binary.json'

    # Check if GPU is available
    if (torch.backends.mps.is_available()) and (torch.backends.mps.is_built()): # type: ignore (pylance confused about torch.backends.mps)
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #device = torch.device("cpu")
    print(f'Using device: {device}')

    # Download dataset from huggingface and preprocess it
    datasets = load_dataset('carblacac/twitter-sentiment-analysis')
    train_df = pd.DataFrame(datasets['train'])
    val_df = pd.DataFrame(datasets['validation']).sample(200)
    X_train = train_df[['text']]
    X_val = val_df[['text']]
    y_train = train_df[['feeling']]
    y_val = val_df[['feeling']]

    # print value counts of y_val
    print('The repartition on validation labels is :', y_val['feeling'].value_counts())

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large', do_lower_case=True)

    # Load model for multi-class classification. num_labels is the number of unique labels in the dataset.
    # with open(label_mapping_path, 'r') as f:
    #     label_mapping = json.load(f)
    # num_labels = len(label_mapping.values())
    num_labels = 2
    print(f'Number of labels: {num_labels}')
    model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large', num_labels=num_labels)

    # Create dataset
    train_dataset = Dataset.from_pandas(X_train)
    val_dataset = Dataset.from_pandas(X_val)

    # Encode text
    train_encodings = tokenizer(train_dataset['text'], truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_dataset['text'], truncation=True, padding=True, max_length=512)

    # Create conversion class

    class BertProcessedDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx]).to(device)
            return item

        def __len__(self):
            return len(self.labels)
        
    # Convert encodings to PyTorch tensors
    train_dataset = BertProcessedDataset(train_encodings, y_train['feeling'].tolist())
    val_dataset = BertProcessedDataset(val_encodings, y_val['feeling'].tolist())

    
    model = model.to(device)

    # Define learning rate
    learning_rate = 2e-5

    # Create checkpoint directory. Name contains date and time of training start.
    now = datetime.now().strftime(f'%Y-%m-%d_%H-%M-AND_lr={str(learning_rate).replace(".", ",")}')
    os.makedirs(f'./checkpoints/{now}', exist_ok=True)
    os.makedirs(f'./checkpoints/{now}/logs', exist_ok=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./checkpoints/{now}',
        num_train_epochs=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./checkpoints/{now}/logs',
        logging_strategy='steps',
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=2000,
        eval_accumulation_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        learning_rate=learning_rate,
        save_strategy='steps',
        save_steps=2000,
        use_mps_device=True,
    )

    # Define compute metrics function
    def compute_metrics(eval_pred):
        accuracy_metric = evaluate.load('accuracy')
        logits, labels = eval_pred
        logits = logits[0] if isinstance(logits, tuple) else logits
        predictions = np.argmax(logits, axis=-1)
        # Calculate accuracy
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
        return {
            'accuracy': accuracy
        }
    
    # Create trainer    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.1)]
    )

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.mps.empty_cache()

    # Train model
    trainer.train()

    # Save model
    trainer.save_model(f'./checkpoints/{now}/model')


if __name__ == '__main__':
    train_model()
