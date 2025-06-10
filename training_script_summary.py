import pandas as pd
import numpy as np
import torch
import argparse

from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers.convert_graph_to_onnx import convert
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class TextClassifierDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float).to(device)
        return item

def train(final_df_path):
    final_df = pd.read_csv(final_df_path)
    test_split = 0.2
    train_df, test_df = train_test_split(
        final_df,
        test_size=test_split,
    )
    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in test set: {len(test_df)}")

    not_chosen_columns = ['titles', 'summaries','terms', "title_and_summary"]
    label_columns = [col for col in final_df.columns if col not in not_chosen_columns]
    df_labels_train = train_df[label_columns]
    df_labels_test = test_df[label_columns]

    labels_list_train = df_labels_train.values.tolist()
    labels_list_test = df_labels_test.values.tolist()

    

    train_texts = train_df['title_and_summary'].tolist()
    train_labels = labels_list_train

    eval_texts = test_df['title_and_summary'].tolist()
    eval_labels = labels_list_test

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=512)
    eval_encodings = tokenizer(eval_texts, padding="max_length", truncation=True, max_length=512)



    train_dataset = TextClassifierDataset(train_encodings, train_labels)
    eval_dataset = TextClassifierDataset(eval_encodings, eval_labels)

    title_model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=len(label_columns),
    )

    title_model = title_model.to(device)

    training_arguments = TrainingArguments(
        output_dir=".",
        dataloader_pin_memory=False,
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
    )

    trainer = Trainer(
        model=title_model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    title_model.save_pretrained("./models/title_summary_model")
    convert(
        framework="pt",
        model="./models/title_summary_model",
        output="./models/title_summary_model.onnx",
        opset=12,
        tokenizer=tokenizer,
    )

if __name__=="__main__":
    # Required positional argument
    parser = argparse.ArgumentParser(description='Arguments for The script')

    parser.add_argument('--clean_dataset', type=str,
                        help='path to the cleaned_dataset csv file')
    args = parser.parse_args()
    train(args.clean_dataset)