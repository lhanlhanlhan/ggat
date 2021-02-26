from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=0, help='Patience')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset to be trained.')
parser.add_argument('--purge_mode', type=str, default='loss',
                    help='Purge history saved dicts if a higher (loss/acc) appears.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

"""Delete all result dictionaries before proceeding the new training process"""
files = glob.glob('*.pkl')
for file in files:
    os.remove(file)

"""Generate data file path and load data """
data_name = args.dataset
data_path = './data/{}/'.format(data_name)
adj, features, labels, idx_train, idx_val, idx_test = load_data(data_path, data_name)

"""Model and optimizer"""
model = GGAT(n_features=features.shape[1],
             n_hid=args.hidden,
             n_class=int(labels.max()) + 1,
             dropout=args.dropout,
             n_heads=args.nb_heads,
             alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item(), acc_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


"""Train model"""
t_total = time.time()
loss_values = []
acc_values = []
bad_counter = 0
best = args.epochs + 1 if args.purge_mode == 'loss' else 0
best_epoch = 0
for epoch in range(args.epochs):
    l, a = train(epoch)
    loss_values.append(l)
    acc_values.append(a)

    """Save training results and purge historical results according to purge method defined (loss/accuracy)"""

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if args.purge_mode == 'loss':
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
    else:
        if acc_values[-1] > best:
            best = acc_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

    if args.patience != 0 and bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

"""Save training results of the last epoch"""
torch.save(model.state_dict(), 'the-last.pkl')

"""Restore the best-ever model"""
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

"""Test the best-ever model"""
compute_test()

"""Restore the model from the last epoch"""
print('Loading the last epoch')
model.load_state_dict(torch.load('the-last.pkl'))

"""Test the model from the last epoch"""
compute_test()
