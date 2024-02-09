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

# def log_results(filename, epoch, train_loss, train_acc, valid_loss, valid_acc):
#     fields = ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']
#     exists = os.path.isfile(filename)
#
#     with open(filename, 'a', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fields)
#         if not exists:
#             writer.writeheader()  # Only write header if file does not exist
#         writer.writerow({
#             'Epoch': epoch,
#             'Train Loss': train_loss,
#             'Train Accuracy': train_acc,
#             'Validation Loss': valid_loss,
#             'Validation Accuracy': valid_acc
#         })

def log_results(filename, epochs, train_loss, train_acc, valid_loss, valid_acc):
    with open(filename, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(["Epochs: " + str(epochs)])
        write.writerows(np.array(train_loss))
        write.writerows(np.array(train_acc))
        write.writerows(np.array(valid_loss))
        write.writerows(np.array(valid_acc))


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

def train_func(sub_train_, model, criterion, optimizer, scheduler, batch_size, device):
    model.train()
    # Train the model
    train_loss = []
    train_acc = []
    data = DataLoader(sub_train_, batch_size=batch_size, shuffle=True,
                      collate_fn=generate_batch)
    for i, (labels, text, lengths) in enumerate(data):
        optimizer.zero_grad()
        text, labels = text.to(device), labels.to(device)
        output = model(text, lengths)
        loss = criterion(output, labels)
        train_loss.append(loss.item() / batch_size)
        loss.backward()
        optimizer.step()
        train_acc.append((output.argmax(1) == labels).sum().item() / batch_size)

        if i % 500 == 0:
            print(f'\tLoss: {np.array(train_loss).mean():.4f} (train)\t|\tAcc: {np.array(train_acc).mean() * 100:.1f}% (train)')

    # Adjust the learning rate
    scheduler.step()

    # Correctly return average loss and accuracy
    return np.array(train_loss), np.array(train_acc)

def test(data_, model, criterion, batch_size, device):
    model.eval()
    test_loss = []
    test_acc = []
    total_samples = 0  # Keep track of total samples for accuracy calculation
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    for i, (labels, text, lengths) in enumerate(data):
        text, labels = text.to(device), labels.to(device)
        with torch.no_grad():
            output = model(text, lengths)
            pred = output.argmax(1)
            loss = criterion(output, labels)
            test_loss.append(loss.item() / batch_size)  # Correct accumulation of loss

            total_samples += labels.size(0)  # Update total samples
            true_pos += ((pred == 1) & (labels == 1)).sum().item()
            true_neg += ((pred == 0) & (labels == 0)).sum().item()
            false_pos += ((pred == 1) & (labels == 0)).sum().item()
            false_neg += ((pred == 0) & (labels == 1)).sum().item()

            test_acc.append((pred == labels).sum().item() / batch_size)

            if i % 500 == 0:
                print(f'\tLoss: {np.array(test_loss).mean():.4f} (test)\t|\tAcc: {np.array(test_acc).mean() * 100:.1f}% (test)')


    acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    # if our dataset only contains labels of 0, we cannot test for precision or recall and therefore f1
    if true_pos + false_neg == 0:
        spec = true_neg / (true_neg + false_pos)
        print("Accuracy: " + str(acc), "    Precision: ----", "    Recall: ----", "    Specificity: "  + str(spec), "    F1 Score: ----")
    # if our dataset only contains labels of 1, we cannot test for specificity
    elif true_neg + false_pos == 0:
        prec = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = (2 * prec * recall) / (prec + recall)
        print("Accuracy: " + str(acc), "    Precision: " + str(prec), "    Recall: " + str(recall), "    Specificity: ----", "    F1 Score: " + str(f1))
    # if our dataset contains a mix of 0 and 1 labels, we can test for all metrics
    else:
        prec = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        spec = true_neg / (true_neg + false_pos)
        f1 = (2 * prec * recall) / (prec + recall)
        print("Accuracy: " + str(acc), "    Precision: " + str(prec), "    Recall: " + str(recall), "    Specificity: "  + str(spec), "    F1 Score: " + str(f1))

    # Correctly return average loss and accuracy
    return np.array(test_loss), np.array(test_acc)

# save a checkpoint if it has the best seen accuracy
def save_checkpoint(model, output_dir):
    torch.save(model, output_dir)



def main(FLAGS):
    # Hyperparameters
    EPOCHS = FLAGS.epochs
    LR = FLAGS.lr 
    BATCH_SIZE = FLAGS.batch_size
    device = torch.device("cuda:" + str(FLAGS.cuda_num) if torch.cuda.is_available() else "cpu")

    # make model directory
    save_dir = FLAGS.model_dir_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #LSTM parameters
    NUM_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 100

    # Load and preprocess dataset
    df = pd.read_csv(FLAGS.data_file)
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
    train_data, test_data, train_labels, test_labels = train_test_split(text_data_padded, labels, test_size = FLAGS.test_split / 100, random_state=42)

    # Creating TensorDataset for train and test datasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    # Define the model, criterion, optimizer, and scheduler
    vocab_size = len(vocab)

    # if evaluating, load trained model
    if FLAGS.evaluate:
        model = torch.load(save_dir + "model_best.pth.tar")
    # else make fresh model
    else:
        model = TextLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 2, NUM_LAYERS, BIDIRECTIONAL, DROPOUT)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    best_acc1 = 0

    # if evaluating,no epochs needed
    if FLAGS.evaluate:
        test(test_dataset, model, criterion, BATCH_SIZE, device)
    # train the model
    else:
        for epoch in tqdm(range(1, EPOCHS + 1), desc = "Epoch: "):
            start_time = time.time()


            # train and test for all epochs
            train_loss_epoch, train_acc_epoch = train_func(train_dataset, model, criterion, optimizer, scheduler, BATCH_SIZE, device)
            valid_loss_epoch, valid_acc_epoch = test(test_dataset, model, criterion, BATCH_SIZE, device)

            # remember best acc@1
            is_best = np.array(train_acc_epoch).mean() * 100 > best_acc1
            best_acc1 = max(np.array(train_acc_epoch).mean() * 100, best_acc1)

            # save the current model if it is better than the best so far
            if is_best:
                save_checkpoint(model, save_dir + "model_best.pth.tar")

            # track loss and acc
            train_loss.append(train_loss_epoch)
            train_acc.append(train_acc_epoch)
            valid_loss.append(valid_loss_epoch)
            valid_acc.append(valid_acc_epoch)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print(f'Epoch: {epoch}, | Elapsed Time: {mins:.2f} minutes, {secs} seconds')
            print(f'\tLoss: {np.array(train_loss_epoch).mean():.4f}(train)\t|\tAcc: {np.array(train_acc_epoch).mean() * 100:.1f}%(train)')
            print(f'\tLoss: {np.array(valid_loss_epoch).mean():.4f}(valid)\t|\tAcc: {np.array(valid_acc_epoch).mean() * 100:.1f}%(valid)')

        # Log results
        log_results("LSTM_training_results.csv", EPOCHS, train_loss, train_acc, valid_loss, valid_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Perform training of LSTM for binary sentence classification.')
    parser.add_argument('--cuda_num',
            type=int, default = 0,
            help='The number of the GPU you want to use.')
    parser.add_argument('--data_file',
            type = str, default = "datasets/funny_dataset.csv",
            help = 'The File for your binary sentence training set.')
    parser.add_argument('--test_split',
            type = int, default = 30,
            help = 'Test split percentage as an integer 0 - 100.')
    parser.add_argument('--batch_size',
            type = int, default = 32,
            help = 'Batch size.')
    parser.add_argument('--print_freq',
            type = int, default = 100,
            help = 'How many batches should pass before a training progress update is printed.')
    parser.add_argument('--lr',
            type = float, default = 1e-3,
            help = 'learning rate xe-y.')
    parser.add_argument('--epochs',
            type = int, default = 2,
            help = 'Number of epochs for training.')
    parser.add_argument('--evaluate', 
            dest='evaluate', action='store_true',
            help='Evaluate a trained model from model_dir_name')
    parser.add_argument('--model_dir_name',
            type = str, default = "LSTM_model/",
            help = 'The directory for the fine-tuned LSTM model.')
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
