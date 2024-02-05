# This code was written with assitance from the tutorial found at https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# and the torch training example found at https://github.com/pytorch/examples/blob/main/imagenet/main.py
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
from enum import Enum
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

seed_value = 42

def fine_tune(tokenizer, model, data, data_labels, FLAGS, device):
    # Determine the maximum sentence length for BERT padding in a batched manner
    print("Determining Max Sentence Length")
    max_len = 0
    for i in tqdm(range(len(data))):
        sentence = data[i]
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        tokenized_data = tokenizer.encode(sentence, add_special_tokens = True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(tokenized_data))

    # add 10 to max_length just in case
    max_len += 10
    print("Max sentence length set to: " + str(max_len))

    # max_len = 70

    # Tokenize all of the sentences and map the tokens to their word IDs.
    print("Tokenize all of the sentence data")
    tokenized_data = []
    attention_masks = []
    num_batches = int(len(data) / FLAGS.batch_size)
    remainder = len(data) % FLAGS.batch_size
    # tokernize data in a batched manner
    for i in tqdm(range(num_batches + 1)):
        if i < num_batches:
            sentences = data[(i * FLAGS.batch_size) : ((i * FLAGS.batch_size) + FLAGS.batch_size)]
        elif i == num_batches and remainder != 0:
            sentences = data[(i * FLAGS.batch_size) : ((i * FLAGS.batch_size) + remainder)]
        else:
            break

        encoded_dict = tokenizer(
            sentences,                    # Sentences to encode
            add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
            max_length = max_len,         # Pad & truncate all sentences
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True, # Construct attention masks
            return_tensors = 'pt',        # Return pytorch tensors
        )
        
        # Add the encoded sentence to the list    
        tokenized_data.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding)
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Convert the lists into pytorch tensors
    tokenized_data = torch.cat(tokenized_data, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(tokenized_data, attention_masks, data_labels)

    # Train test split
    if FLAGS.test_split == 100:
        val_dataset = dataset
    else:
        train_percentage = (100 - FLAGS.test_split) / 100
        train_size = int(train_percentage * len(dataset))
        val_size = len(dataset) - train_size
        # Generator for random split so that it is consistent across training attempts
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = torch.Generator().manual_seed(seed_value))

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order. 
        train_loader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = FLAGS.batch_size # Trains with this batch size.
        )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    val_loader = DataLoader(
        val_dataset, # The validation samples.
        sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
        batch_size = FLAGS.batch_size # Evaluate with this batch size.
    )

    # if we are evaluating a locally stored trained model
    if FLAGS.evaluate: 
        # evaluate on the validation set
        validate(val_loader, model, FLAGS.print_freq, FLAGS.evaluate, device)
    # if we are fine-tuning from the pretrained huggingface model
    else:
        # AdamW optimizer from the huggingface library
        optimizer = torch.optim.AdamW(model.parameters(),
            lr = FLAGS.lr,
            eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
        )

        # Total number of training steps is [number of batches] x [number of epochs]. 
        total_steps = len(train_loader) * FLAGS.epochs
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        # Make dir to save trained modelif needed
        output_dir = FLAGS.model_dir_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # train and validatde for all epochs, 
        best_acc1 = 0
        train_loss_tracker = []
        train_acc_tracker = []
        val_loss_tracker = []
        val_acc_tracker = []

        for epoch in tqdm(range(FLAGS.epochs)):
            # train for an epoch
            train_l, train_a = train(train_loader, model, optimizer, scheduler, epoch, FLAGS.print_freq, device)

            train_loss_tracker.append(np.array(train_l))
            train_acc_tracker.append(np.array(train_a))

            # evaluate on the validation set
            acc1, val_l, val_a = validate(val_loader, model, FLAGS.print_freq, FLAGS.evaluate, device)

            val_loss_tracker.append(np.array(val_l))
            val_acc_tracker.append(np.array(val_a))

            # remember best acc@1
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            # save the current model if it is better than the best so far
            save_checkpoint(model, tokenizer, FLAGS.model_dir_name, is_best)

        # write the loss and acc values to a csv file
        # save the noise ignorance test
        with open("BERT_training_results.csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(["Epochs: " + str(FLAGS.epochs), "Train Batches: " + str(len(train_loader)), "Validation Batches: " + str(len(val_loader))])
            write.writerows(np.array(train_loss_tracker))
            write.writerows(np.array(train_acc_tracker))
            write.writerows(np.array(val_loss_tracker))
            write.writerows(np.array(val_acc_tracker))

    return

def train(train_loader, model, optimizer, scheduler, epoch, print_freq, device):
    # track data
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    loss_tracker = []
    acc_tracker = []

    loop_end = time.time()
    update_end = time.time()
    for i, (tokens, masks, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - loop_end)

        # Put batch of data on device. 
        tokens = tokens.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # Perform a forward pass
        loss, logits = model(tokens, token_type_ids = None, attention_mask = masks, labels = labels, return_dict = False)

        # measure accuracy and record loss
        acc1 = accuracy(logits, labels, topk=(1,))
        losses.update(loss.item(), tokens.size(0))
        top1.update(acc1[0].item(), tokens.size(0))

        loss_tracker.append(loss.item())
        acc_tracker.append(acc1[0].item())

        # Clear gradients before a new backward pass
        optimizer.zero_grad()        
        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0 to help prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

        loop_end = time.time()

        if i != 0 and i % print_freq == 0:
            # measure elapsed time since the last update
            batch_time.update(time.time() - update_end)
            update_end = time.time()

            progress.display(i + 1)

    return loss_tracker, acc_tracker

def validate(val_loader, model, print_freq, evaluate, device):
    def run_validate(loader, base_progress = 0):
        if evaluate:
            true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

        with torch.no_grad():
            end = time.time()

            loss_tracker = []
            acc_tracker = []

            for i, (tokens, masks, labels) in enumerate(loader):
                i = base_progress + i
                
                # Put batch of data on device. 
                tokens = tokens.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                # Perform a forward pass
                loss, logits = model(tokens, token_type_ids = None, attention_mask = masks, labels = labels, return_dict = False)

                # measure accuracy and record loss
                acc1 = accuracy(logits, labels, topk=(1,))
                losses.update(loss.item(), tokens.size(0))
                top1.update(acc1[0].item(), tokens.size(0))

                loss_tracker.append(loss.item())
                acc_tracker.append(acc1[0].item())

                if i != 0 and  i % print_freq == 0:
                    progress.display(i + 1)

                if evaluate:
                    _, pred = logits.topk(1, 1, True, True)
                    pred = pred.t()[0]

                    for j in range(len(labels)):
                        # not funny
                        if labels[j] == 0:
                            if pred[j] == 0:
                                true_neg += 1
                            elif pred[j] == 1:
                                false_pos += 1
                        # funny
                        elif labels[j] == 1:
                            if pred[j] == 0:
                                false_neg += 1
                            elif pred[j] == 1:
                                true_pos += 1

                # measure elapsed t ime
                batch_time.update(time.time() - end)
                end = time.time()

        if evaluate:
            try:
                acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
                prec = true_pos / (true_pos + false_pos)
                recall = true_pos / (true_pos + false_neg)
                f1 = (2 * prec * recall) / (prec + recall)
                # print("Classification Accuracy: " + str(top1))
                print("Accuracy: " + str(acc), "    Precision: " + str(prec), "   Recall: " + str(recall), "    F1 Score: " + str(f1))
            except(ZeroDivisionError):
                print("Dataset unsuitable for precision and recall, it only has false classes.")


        return loss_tracker, acc_tracker

    # track data
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    loss_tracker, acc_tracker = run_validate(val_loader)
    progress.display_summary()

    return top1.avg, loss_tracker, acc_tracker

# save a checkpoint if it has the best seen accuracy
def save_checkpoint(model, tokenizer, output_dir, is_best):
    if is_best:
        model_to_save = model.module if hasattr(model, 'module') else model # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

# used to print training and validation tracking information
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

# used to print training and validation tracking information
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'

        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

# used to print training and validation tracking information
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# measure how accurate the model is over a batch
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(FLAGS):
    # set the device
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    # load the dataset
    df = pd.read_csv(FLAGS.data_file, delimiter = ',', dtype={'label': bool}, header = 1, names = ['sentence', 'label'])

    # Get the sentence data and the labels into numpy arrays and discard the first row as it contains the headers
    # make the data a list of sentences
    data = df.sentence.values.tolist()
    # make labels binary 0, 1 from bool
    labels = torch.tensor(df.label.values, dtype = int)

    # Load the BERT tokenizer
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

    # Load the BERT sequence classifier model
    print('Loading BERT model...')

    # if we are evaluating a local trained model
    if FLAGS.evaluate:
        model = BertForSequenceClassification.from_pretrained(
            FLAGS.model_dir_name, # BERT model with uncased vocab.
            num_labels = 2, # binary classification.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False # Whether the model returns all hidden-states.
        )
    # if we are fine-tuning from the pretrained huggingface model
    else:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # BERT model with uncased vocab.
            num_labels = 2, # binary classification.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False # Whether the model returns all hidden-states.
        )

    # put model on specified device
    model.to(device)

    fine_tune(tokenizer, model, data, labels, FLAGS, device)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Perform fine tuning of BERT binary sentence classification.')
    parser.add_argument('--cuda_num',
            type=int, default = 0,
            help='The number of the GPU you want to use.')
    parser.add_argument('--data_file',
            type = str, default = "funny_dataset.csv",
            help = 'The File for your binary sentence training set.')
    parser.add_argument('--test_split',
            type = int, default = 30,
            help = 'Test split percentage as an integer 0 - 100.')
    parser.add_argument('--batch_size',
            type = int, default = 32,
            help = '16 or 32 is recommended by BERT authors for fine tuning.')
    parser.add_argument('--print_freq',
            type = int, default = 100,
            help = 'How many batches should pass before a training progress update is printed.')
    parser.add_argument('--lr',
            type = float, default = 5e-5,
            help = '5e-5, 3e-5, 2e-5 is recommended by BERT authors for fine tuning.')
    parser.add_argument('--epochs',
            type = int, default = 2,
            help = '2, 3, 4 is recommended by BERT authors for fine tuning.')
    parser.add_argument('--evaluate', 
            dest='evaluate', action='store_true',
            help='Evaluate a trained model from model_dir_name')
    parser.add_argument('--model_dir_name',
            type = str, default = "BERT_model/",
            help = 'The directory for the fine-tuned BERT model.')
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
