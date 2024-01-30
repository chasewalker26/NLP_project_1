# This code was written with assitance from the tutorial found at https://mccormickml.com/2019/07/22/BERT-fine-tuning/

import os
import time
import torch
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

seed_value = 42

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(tokenizer, model, data, data_labels, FLAGS, device):
    # Determine the maximum sentence length for BERT padding
    max_len = 0
    for sentence in data:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        tokenized_data = tokenizer.encode(sentence, add_special_tokens = True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(tokenized_data))

    # add 10 to max_length just in case
    max_len += 10

    # Tokenize all of the sentences and map the tokens to their word IDs.
    tokenized_data = []
    attention_masks = []
    for sentence in data:
        # `encode_plus` will:
        #   (1) Tokenize the sentence
        #   (2) Prepend the `[CLS]` token and append the `[SEP]` token
        #   (4) Map tokens to their IDs
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens
        encoded_dict = tokenizer.encode_plus(
                            sentence,                     # Sentence to encode
                            add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                            max_length = max_len,         # Pad & truncate all sentences
                            pad_to_max_length = True,
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
    data_labels = torch.tensor(data_labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(tokenized_data, attention_masks, data_labels)

    # Train test split
    train_percentage = 0.70
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

    # AdamW optimizer from the huggingface library
    optimizer = AdamW(model.parameters(),
                    lr = FLAGS.lr,
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    # Total number of training steps is [number of batches] x [number of epochs]. 
    total_steps = len(train_loader) * FLAGS.epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, FLAGS.epochs):
        
        # ========================================
        #               Training
        # ========================================
        print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, FLAGS.epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for step, (tokens, masks, labels) in enumerate(train_loader):

            # Progress update every 32 batches.
            if step % 32 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            # Put batch of data on device. 
            tokens = tokens.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # Claer gradients before a new backward pass
            model.zero_grad()        
            # Perform a forward pass
            loss, logits = model(tokens,  token_type_ids = None,  attention_mask = masks,  labels = labels)
            # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end.
            total_train_loss += loss.item()
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

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)            
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        print("\nRunning Validation...")

        t0 = time.time()

        # Put the model in evaluation mode
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for step, (tokens, masks, labels) in enumerate(val_loader):
            
            # Put batch of data on device. 
            tokens = tokens.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                (loss, logits) = model(tokens, token_type_ids = None, attention_mask = masks, labels = labels)
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = labels.cpu().numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_loader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_loader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # Display floats with three decimal places
    pd.set_option('precision', 3)

    # Print training statistics.
    df_stats = pd.DataFrame(data = training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

    # save the trained model to a directory
    output_dir = FLAGS.path_to_save_model
    print("Saving model to %s" % output_dir)
    # Create directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return

def main(FLAGS):
    # set the device
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    # load the dataset
    df = pd.read_csv(FLAGS.path_to_data, delimiter = '\t', header = None, names = ['sentence', 'label'])

    # Get the sentence data and the labels into numpy arrays
    data = df.sentence.values
    labels = df.label.values

    # Load the BERT tokenizer
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

    # Load the BERT sequence classifier model
    print('Loading BERT model...')
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # BERT model with uncased vocab.
        num_labels = 2, # binary classification.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False # Whether the model returns all hidden-states.
    )
    # put model on specified device
    model.to(device)

    train(tokenizer, model, device, data, labels, FLAGS, device)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Perform fine tuning of BERT for binary sentence classification.')
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--path_to_data',
                type = str, default = "funny_dataset",
                help = 'The path to your binary sentence training set.')
    parser.add_argument('--batch_size',
            type = int, default = 32,
            help = '16 or 32 is recommended by BERT authors for fine tuning.')
    parser.add_argument('--lr',
            type = float, default = 2e-5,
            help = '5e-5, 3e-5, 2e-5 is recommended by BERT authors for fine tuning.')
    parser.add_argument('--epochs',
            type = int, default = 4,
            help = '2, 3, 4 is recommended by BERT authors for fine tuning.')
    parser.add_argument('--path_to_save_model',
            type = str, default = "BERT_model",
            help = 'The path to save the fine-tuned BERT model.')
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)