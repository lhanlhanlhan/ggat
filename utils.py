import numpy as np
import scipy.sparse as sp
import torch
import random


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def split_training_samples(idx_list):
    """Shuffle"""
    n = len(idx_list)
    random.shuffle(idx_list)
    end_train = int(n * 0.4)
    end_val = int(n * 0.8)

    """Split training samples"""
    idx_train = idx_list[:end_train - 1]
    idx_val = idx_list[end_train:end_val - 1]
    idx_test = idx_list[end_val:]

    return idx_train, idx_val, idx_test


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (default: cora)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    if dataset.startswith("KIRC"):
        """KIRC"""
        # 先构造数字 0 ~ 609
        idx_list = list(range(610))
        for _ in range(6):
            idx_list.extend(list(range(538, 610)))

        idx_train, idx_val, idx_test = split_training_samples(idx_list)

    elif dataset.startswith("LIHC"):
        """LIHC"""
        idx_list = list(range(421))
        for _ in range(6):
            idx_list.extend(list(range(371, 421)))

        idx_train, idx_val, idx_test = split_training_samples(idx_list)

    elif dataset.startswith("LUAD"):
        """LUAD"""
        idx_list = list(range(592))
        for _ in range(6):
            idx_list.extend(list(range(533, 592)))

        idx_train, idx_val, idx_test = split_training_samples(idx_list)

    elif dataset.startswith("cancer_all"):
        """cancer_all"""
        idx_list = list(range(1442))

        idx_train, idx_val, idx_test = split_training_samples(idx_list)

    else:
        """CORA: no shuffle"""
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
