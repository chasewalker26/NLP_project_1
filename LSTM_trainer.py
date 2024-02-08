import os
import csv
import time
import torch
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import csv

def log_results(filename, epoch, train_loss, train_acc, valid_loss, valid_acc):
    fields = ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']
    exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        if not exists:
            writer.writeheader()  # Only write header if file does not exist
        writer.writerow({
            'Epoch': epoch,
            'Train Loss': train_loss,
            'Train Accuracy': train_acc,
            'Validation Loss': valid_loss,
            'Validation Accuracy': valid_acc
        })


class TextLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super(TextLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.encoder = torch.nn.LSTM(embedding_dim,
                                     hidden_dim,
                                     num_layers=n_layers,
                                     bidirectional=bidirectional,
                                     dropout=dropout,
                                     batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        # Pack sequence
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.encoder(packed_embedded)
        # Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        if hidden.shape[0] > 1:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden.squeeze(0))
        return self.fc(hidden)

def generate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
         label_list.append(_label)
         processed_text = torch.tensor(_text, dtype=torch.int64)
         text_list.append(processed_text)
         lengths.append(processed_text.size(0))
    return torch.tensor(label_list, dtype=torch.int64), pad_sequence(text_list, batch_first=True), torch.tensor(lengths, dtype=torch.int64)

def train_func(sub_train_):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (labels, text, lengths) in enumerate(data):
        optimizer.zero_grad()
        text, labels = text.to(device), labels.to(device)
        output = model(text, lengths)
        loss = criterion(output, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == labels).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for labels, text, lengths in data:
        text, labels = text.to(device), labels.to(device)
        with torch.no_grad():
            output = model(text, lengths)
            pred = output.argmax(1)
            loss = criterion(output, labels)
            loss += loss.item()
            acc += (pred == labels).sum().item()
            true_pos += ((pred == 1) & (labels == 1)).sum().item()
            true_neg += ((pred == 0) & (labels == 0)).sum().item()
            false_pos += ((pred == 1) & (labels == 0)).sum().item()
            false_neg += ((pred == 0) & (labels == 1)).sum().item()

    # Evaluation metrics calculation
    try:
        acc_metric = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        prec = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = (2 * prec * recall) / (prec + recall)
        print(f"Accuracy: {acc_metric:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    except ZeroDivisionError:
        print("Dataset unsuitable for precision and recall, it only has false classes.")

    return loss / len(data_), acc / len(data_)

if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 25 
    LR = 1e-3 
    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #LSTM parameters
    NUM_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 100

    # Load and preprocess dataset
    df = pd.read_csv("datasets/funny_dataset.csv")
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, df['text']), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    def data_process(line):
        return [vocab[token] for token in tokenizer(line)]

    # Convert processed text data into a tensor where each row is a sequence of indices
    df['text'] = df['text'].apply(data_process)
    from torch.nn.utils.rnn import pad_sequence

    # Convert lists of token indices into tensors and pad sequences
    text_data = [torch.tensor(text_sequence, dtype=torch.int64) for text_sequence in df['text']]
    text_data_padded = pad_sequence(text_data, batch_first=True, padding_value=vocab['<pad>'])

    labels = torch.tensor(df['humor'].values, dtype=torch.int64)

    # Split dataset
    train_data, test_data, train_labels, test_labels = train_test_split(text_data_padded, labels, test_size=0.25, random_state=42)

    # Creating TensorDataset for train and test datasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    # Define the model, criterion, optimizer, and scheduler
    vocab_size = len(vocab)
    model = TextLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 2, NUM_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    filename = "LSTM_training_results.csv"

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss, train_acc = train_func(train_dataset)
        valid_loss, valid_acc = test(test_dataset)

        # Log results
        log_results(filename, epoch, train_loss, train_acc, valid_loss, valid_acc)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print(f'Epoch: {epoch}, | time in {mins} minutes, {secs} seconds')
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
