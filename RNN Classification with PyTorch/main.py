import os
import re
import sys
import time
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, LabelField, TabularDataset, Pipeline, BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence

"""
# RNN Classification with PyTorch
"""

def seed_reset(SEED=0):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def _download_dataset(size=10000):
    assert sys.version_info >= (3, 6), "Use Python3.6+"

    import ssl
    import urllib.request
    url = "https://raw.githubusercontent.com/dongkwan-kim/small_dataset/master/review_{}k.csv".format(size // 1000)

    dir_path = "../data"
    file_path = os.path.join(dir_path, "review_{}k.csv".format(size // 1000))
    if not os.path.isfile(file_path):
        print("Download: {}".format(file_path))
        os.makedirs(dir_path, exist_ok=True)
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as u, open(file_path, 'wb') as f:
            f.write(u.read())
    else:
        print("Already exist: {}".format(file_path))


def _load_dataset(test_data_path=None, size=10000, train_test_ratio=0.8, seed=0):
    _download_dataset()

    preprocess_pipeline = Pipeline(lambda x: re.sub(r'[^a-z]+', ' ', x))

    TEXT = Field(batch_first = True,
                include_lengths = True, 
                lower=True, 
                preprocessing=preprocess_pipeline)
    LABEL = LabelField(dtype = torch.float)

    train_data = TabularDataset(path="../data/review_{}k.csv".format(size // 1000), 
                                format='csv', 
                                fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)
    test_data = None
    if test_data_path is not None:
        test_data = TabularDataset(path=test_data_path, 
                                    format='csv', 
                                    fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)
    
    
    train_data, valid_data = train_data.split(split_ratio=train_test_ratio, 
                                            random_state = random.seed(seed))
    
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    vocab_size = len(TEXT.vocab)
    padding_idx = TEXT.vocab.stoi[TEXT.pad_token]

    return train_data, valid_data, test_data, vocab_size, padding_idx


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def accuracy(prediction, label):
    binary_prediction = torch.round(torch.sigmoid(prediction))
    correct = (binary_prediction == label).float()
    acc = correct.sum() / len(correct)
    return acc


class RNN(nn.Module):
    def __init__(self, num_embeddings, padding_idx, embedding_dim, 
                 hidden_dim, num_layers, dropout, bidirectional):
        """ RNN model

        :param num_embeddings: the numebr of embeddings (vocab size)
        :param padding_idx: padding idx
        :param embedding_dim: (int) embedding dimension
        :param hidden_dim: (int) hidden dimension
        :param num_layers: (int) the number of recurrent layers
        :param dropout: (float) dropout rate
        :param bidirectional: (bool) is bidirectional

        :return output: type=torch.Tensor, shape=[batch size]
        """
        
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx = padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional        

        self.lstm = True
        self.rnn: nn.Module = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, 
                                        num_layers = num_layers, batch_first = True, 
                                        bidirectional = bidirectional)   

        num_directions = 2 if bidirectional else 1
        self.fc: nn.Module = nn.Linear(hidden_dim * num_directions, 1)

        
    def forward(self, text, text_lengths):
        
        ## text.shape = [batch size, max text length, embedding_dim]
        ## text_lengths.shape = [batch_size]
 
        embedded = self.dropout(self.embedding(text))
        packed_embedded = pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first = True)

        output = self.rnn(packed_embedded)[1][0] if self.lstm else self.rnn(packed_embedded)[1]
        output = torch.cat((output[-2], output[-1]), 1) if self.bidirectional else output[-1:, :, :]
        output = self.fc(output).flatten()

        assert output.shape == torch.Size([text.shape[0]]) # batch_size
        
        return output


def train(model, iterator, optimizer, criterion):
    """ train method
    :param model: RNN model
    :param iterator: train dataset iterator
    :param optimizer: optimzer
    :param criterion: loss function

    :return output: train loss, train accuracy
    """
    
    total_epoch_loss = 0
    total_epoch_acc = 0
    
    model.train()
    for batch in tqdm(iterator, desc="train"):

        optimizer.zero_grad()
        (text, text_lengths), labels = batch.review, batch.sentiment
        
        prediction = model(text, text_lengths)
        loss:torch.Tensor = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
        acc:torch.Tensor = accuracy(prediction, labels)

        assert loss.shape == torch.Size([])
        assert acc.shape == torch.Size([])
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
    
    epoch_loss = total_epoch_loss / len(iterator)
    epoch_acc = total_epoch_acc / len(iterator)
    return epoch_loss, epoch_acc


def evaluate(model, iterator, criterion):
    """ evaluation method
    :param model: RNN model
    :param iterator: dataset iterator
    :param criterion: loss function

    :return output: loss, accuracy
    """
    
    total_epoch_loss = 0
    total_epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator, desc="evaluate"):
            (text, text_lengths), labels = batch.review, batch.sentiment

            prediction = model(text, text_lengths)
            loss:torch.Tensor = criterion(prediction, labels)
            acc:torch.Tensor = accuracy(prediction, labels)

            assert loss.shape == torch.Size([])
            assert acc.shape == torch.Size([])

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    epoch_loss = total_epoch_loss / len(iterator)
    epoch_acc = total_epoch_acc / len(iterator)
    return epoch_loss, epoch_acc


def set_hyperparameter_dict():
    param_dict = {
        'embedding_dim': 150,
        'hidden_dim': 400,
        'num_layers': 2,
        'dropout': 0.6,
        'bidirectional': True, 
        'batch_size': 32,
        'num_epochs': 17,
        'learning_rate': 1e-3,
        'device':'cuda'
    }
    return param_dict


def run(num_samples=10000, param_dict=set_hyperparameter_dict(), train=train, evaluate=evaluate, seed=0, test_data_path=None, verbose=True):

    train_data, valid_data, test_data, vocab_size, padding_idx = _load_dataset(test_data_path, num_samples, seed=seed)
    
    NUM_EMBEDDINGS = vocab_size
    PADDING_IDX = padding_idx

    param_dict = set_hyperparameter_dict()

    model = RNN(NUM_EMBEDDINGS, 
                PADDING_IDX,
                param_dict['embedding_dim'], 
                param_dict['hidden_dim'], 
                param_dict['num_layers'], 
                param_dict['dropout'], 
                param_dict['bidirectional']
                )

    device = torch.device(param_dict['device'] if torch.cuda.is_available() else 'cpu')

    train_iter, val_iter = BucketIterator.splits(
                                    (train_data, valid_data), 
                                    batch_size = param_dict['batch_size'],
                                    sort_within_batch = True,
                                    sort_key=lambda x: len(x.review),
                                    device = device)
    
    if test_data is not None:
        test_iter = BucketIterator(test_data, 
                                    batch_size = param_dict['batch_size'],
                                    sort_within_batch = True,
                                    sort_key=lambda x: len(x.review),
                                    device = device)

    train_loss, train_acc = None, None
    valid_loss, valid_acc = None, None
    test_loss, test_acc = None, None

    optimizer = optim.Adam(model.parameters(), lr=param_dict['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(param_dict['num_epochs']):
        print(f'Epoch: {epoch+1:02}')
        start_time = time.time()

        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_iter, criterion)   
                   
        if verbose:
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'\nEpoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    if test_data is not None:
        test_loss, test_acc = evaluate(model, test_iter, criterion)
        print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

    return train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc


if __name__ == '__main__':
    seed_reset()
    run()