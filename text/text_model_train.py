import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Paths are resolved relative to this file's own location, not the
# current working directory, so training works the same whether you run
# `python text_model_train.py` from this folder or from anywhere else —
# and so it matches text_inference.py's flat-layout MODEL_DIR exactly.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(THIS_DIR, "data", "robot_commands.csv")
MODEL_OUTPUT = os.path.join(THIS_DIR, "models", "text_intent_model")

MAX_LEN = 64

BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 2e-5

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"No training CSV found at {CSV_PATH}. Create a 'data/robot_commands.csv' "
        f"file (columns: text, intent) next to this script."
    )

df = pd.read_csv(CSV_PATH)

print(df.head())

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["intent"])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
)

class RobotDataset(Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):

        return len(self.labels)

train_dataset = RobotDataset(
    train_encodings,
    train_labels,
)

val_dataset = RobotDataset(
    val_encodings,
    val_labels,
)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(encoder.classes_),
)

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions)
    }

training_args = TrainingArguments(

    output_dir=os.path.join(THIS_DIR, "models", "checkpoints"),

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_steps=10,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    greater_is_better=True,
)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )
    ],
)

trainer.train()

results = trainer.evaluate()

print(results)

os.makedirs(MODEL_OUTPUT, exist_ok=True)

trainer.save_model(MODEL_OUTPUT)

tokenizer.save_pretrained(MODEL_OUTPUT)

with open(
    os.path.join(MODEL_OUTPUT, "label_mapping.json"),
    "w",
) as f:

    json.dump(
        {
            str(i): label
            for i, label in enumerate(encoder.classes_)
        },
        f,
        indent=4,
    )

