"""
Script for training a simple MLP for classification on the MNIST dataset
"""
import argparse
import os
import sys
import numpy as np
import random
import pickle
import json
from tqdm import tqdm
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from neural_nets import MLP, ConvNet
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, balanced_accuracy_score

torch.multiprocessing.set_sharing_strategy('file_system')

def output_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    youden = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Youden']
    print (('{:>11}'*len(metrics)).format(*metrics))
    print ((' {:.8f}'*len(metrics)).format(accuracy, precision, recall, f1, youden))


class PacketDataset(Dataset):
    def __init__(self, data, labels, categories):
        self.data = data
        self.labels = labels
        self.categories = categories
        assert(len(self.data) == len(self.labels) == len(self.categories))

    def __getitem__(self, index):
        data, labels, categories = torch.FloatTensor(self.data[index]), torch.FloatTensor(self.labels[index]), torch.FloatTensor(self.categories[index])
        return data, labels, categories

    def __len__(self):
        return len(self.data)

class PacketFlowDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert(len(self.data) == len(self.labels))

    def __getitem__(self, index):
        data, labels = torch.FloatTensor(self.data[index]), torch.FloatTensor(self.labels[index])
        return data, labels

    def __len__(self):
        return len(self.data)

# def train_epoch(model:nn.Module, data_loader:DataLoader, optimizer:Adam, loss_fn:nn.BCELoss):
def train_epoch(model:nn.Module, data_loader:DataLoader, optimizer:Adam, loss_fn:nn.CrossEntropyLoss):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        optimizer (Adam)
        loss_fn (nn.CrossEntropyLoss)

    Returns:
        [Float]: average training loss on epoch
    """
    model.train(mode=True)
    num_batches = len(data_loader)

    loss = 0
    for x, y in data_loader:
        optimizer.zero_grad()
        y = y.squeeze().long()
        logits = model(x)
        batch_loss = loss_fn(logits, y)

        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
    return loss / num_batches


# def eval_epoch(model: nn.Module, data_loader:DataLoader, loss_fn:nn.BCELoss):
def eval_epoch(model: nn.Module, data_loader:DataLoader, loss_fn:nn.CrossEntropyLoss):
    """
    Evaluate epoch on validation data
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        loss_fn (nn.CrossEntropyLoss)

    Returns:
        [Float]: average validation loss 
    """
    model.eval()
    num_batches = len(data_loader)

    loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            pred_y = model(x)
            y = y.squeeze().long()
            batch_loss = loss_fn(pred_y, y)
            loss += batch_loss.item()
    return loss / num_batches

def get_nth_split(dataset, n_fold, index):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
    train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
    return train_indices, test_indices

MULTIPLIER = 2 ** 16
def get_dataset(data, is_binary=True, scaler=None, return_scaler=False):
    data_list = []
    for item in data:
        data_list.append(item)

    new_dataset = []
    new_labels = []

    new_dataset2 = []
    for item, y, y2 in tqdm(data_list):
        item = item.numpy()
        y = y.numpy().astype(np.int64)
        y2 = y2.numpy().astype(np.int64)

        # average = np.zeros(3, dtype=np.int64)
        # deviation = np.zeros(3, dtype=np.int64)
        average = np.zeros(3)
        deviation = np.zeros(3)
        for i in range(item.shape[0]):
            # item[i,:6]: sport, dport, protocol, tot_len, interval, direction
            current_vector = item[i,:6]

            average += current_vector[3:]
            # current_average = (average/(i+1)).astype(np.int64)
            current_average = (average/(i+1))

            deviation += np.abs(current_vector[3:]-current_average)
            current_deviation = (deviation/(i+1))
            # current_deviation = (deviation/(i+1)).astype(np.int64)

            # final_vector = np.concatenate((current_vector, current_average))
            final_vector = np.concatenate((current_vector, current_average, current_deviation))
        new_dataset.append(final_vector)
        # new_dataset2.append(deepcopy(final_vector))

        if is_binary:
            new_labels.append(y[0])
        else:
            if y2[0] == 0:
                _y = np.ones(1) * 0
            elif y2[0] == 1 or y2[0] == 2:
                _y = np.ones(1) * 1
            elif y2[0] >= 3 and y2[0] <= 8:
                _y = np.ones(1) * 2
            elif y2[0] == 9:
                _y = np.ones(1) * 3
            elif y2[0] == 10:
                _y = np.ones(1) * 4
            elif y2[0] == 11 or y2[0] == 12:
                _y = np.ones(1) * 5
            elif y2[0] == 13 or y2[0] == 14:
                _y = np.ones(1) * 6
            else:
                _y = np.ones(1) * 0
            new_labels.append(_y.astype(np.int64))
        # new_dataset.append((final_vector, y[0]))
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(new_dataset)
    new_dataset = scaler.transform(new_dataset)
    # print(scaler.scale_, scaler.data_min_, scaler.data_max_)
    # for i, d in enumerate(new_dataset2):
    #     print(d, ((d - scaler.data_min_)*scaler.scale_*2**16).astype(np.int64), new_dataset[i])
    if return_scaler:
        return new_dataset, new_labels, scaler
    return new_dataset, new_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training a model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden_sizes', help='hidden layer dimensions', nargs='+', type=int, default=[16, 16])
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=10)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.8)
    parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='../dataset')
    # parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='../data')
    parser.add_argument('--save_dir', help='save directory', default='../saved_models', type=Path)
    parser.add_argument('--maxLength', type=int, default=100, help='max length')
    parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
    parser.add_argument('--fold', type=int, default=0, help='fold to use')
    parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
    parser.add_argument('--maxSize', type=int, default=sys.maxsize, help='limit of samples to consider')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--is-binary', action='store_true')

    args = parser.parse_args()
    print(args.hidden_sizes)
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with open (os.path.join(args.data_dir, 'flows.pickle'), "rb") as f:
        all_data = pickle.load(f)

    with open(os.path.join(args.data_dir, "flows_categories_mapping.json"), "r") as f:
        categories_mapping_content = json.load(f)

    categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
    assert min(mapping.values()) == 0

    all_data = [item[:args.maxLength,:] for item in all_data if np.all(item[:,4]>=0)]
    # all_data = [item[:args.maxLength,:] for item in all_data[:10000] if np.all(item[:,4]>=0)]
    random.shuffle(all_data)
    # print("lens", [len(item) for item in all_data])
    x = [item[:, :-2] for item in all_data]
    y = [item[:, -1:] for item in all_data]
    categories = [item[:, -2:-1] for item in all_data]

    dataset = PacketDataset(x, y, categories)

    # split training data to train/validation
    split_r = args.train_val_split

    # globals()[args.function]()
    n_fold = args.nFold
    fold = args.fold

    train_indices, test_indices = get_nth_split(dataset, n_fold, fold)
    train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)

    new_dataset_train, new_labels_train, scaler = get_dataset(train_data, args.is_binary, return_scaler=True)
    new_dataset_test, new_labels_test = get_dataset(test_data, args.is_binary, scaler=scaler)
    train_trainset = PacketFlowDataset(new_dataset_train, new_labels_train)
    test_testset = PacketFlowDataset(new_dataset_test, new_labels_test)
    train_trainset, train_valset = random_split(train_trainset, [round(len(train_trainset)*split_r), round(len(train_trainset)*(1 - split_r))], generator=torch.Generator().manual_seed(SEED))

    # new_dataset, new_labels = get_dataset(dataset)
    # dataset = PacketFlowDataset(new_dataset, new_labels)
    # train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels, random_state=SEED)
    # test_testset = torch.utils.data.Subset(dataset, test_indices)
    # train_indices, val_indices = train_test_split(train_indices, test_size=0.2, stratify=np.array(dataset.labels)[train_indices], random_state=SEED)
    # train_trainset = torch.utils.data.Subset(dataset, train_indices)
    # train_valset = torch.utils.data.Subset(dataset, val_indices)


    in_dim = 12
    if args.is_binary:
        out_dim = 2
    else:
        out_dim = 7
    print(in_dim, out_dim)
    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden_sizes=args.hidden_sizes)

    optimizer = Adam(model.parameters(), lr=1e-3)

    loss_fnc = nn.CrossEntropyLoss()
    # loss_fnc = nn.BCELoss()

    train_loader = DataLoader(train_trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(train_valset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_testset, batch_size=1, shuffle=False)
    print('Training')
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fnc)
        val_loss = eval_epoch(model, val_loader, loss_fnc)
        print(f"Epoch: {epoch  + 1} - train loss: {train_loss:.5f} validation loss: {val_loss:.5f}")

    print('Evaluate model on test data')
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        acc = 0
        for samples, labels in tqdm(test_loader):
            logits = model(samples.float())
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            acc += (preds == labels).sum()
            all_labels += [label for label in labels.detach().numpy().copy()]
            all_preds += [pred for pred in preds.detach().numpy().copy()]

    print(f"Accuracy: {(acc / len(test_testset))*100.0:.3f}%")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict(),
                'hidden_sizes': args.hidden_sizes,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_acc': acc,
                'scaler': scaler},
                f"{args.save_dir}/mlp_pktflw.th")
    all_labels = np.array(all_labels).squeeze()
    all_preds = np.array(all_preds)
    print(classification_report(all_labels, all_preds, digits=6))

    # print(classification_report(all_labels, all_preds, labels=[0, 1]))
    # print(output_scores(all_labels, all_preds))
