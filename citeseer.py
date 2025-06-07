import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp

def load_custom_citeseer(node_path, link_path):
    node_ids = []
    features = []
    labels = []
    with open(node_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_ids.append(parts[0])
            features.append([float(x) for x in parts[1:-1]])
            labels.append(parts[-1])
    node_ids = np.array(node_ids)
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)
    classes = {c: i for i, c in enumerate(sorted(set(labels)))}
    labels = np.array([classes[x] for x in labels], dtype=np.int64)

    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    edges = []
    with open(link_path, 'r') as f:
        for line in f:
            src, dst = line.strip().split()
            if src in id2idx and dst in id2idx:
                edges.append((id2idx[src], id2idx[dst]))
    edges = np.array(edges, dtype=np.int32)

    N = len(node_ids)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
                        shape=(N, N), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return features, labels, adj

def normalize_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.spmm(adj, x)
        return self.linear(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid, dropout)
        self.gc2 = GCNLayer(nhid, nclass, dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    node_path = '/content/citeseer/node'
    link_path = '/content/citeseer/link'

    features, labels, adj = load_custom_citeseer(node_path, link_path)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)

    n = features.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.2 * n)
    n_val = int(0.2 * n)
    idx_train = torch.LongTensor(idx[:n_train]).to(device)
    idx_val = torch.LongTensor(idx[n_train:n_train + n_val]).to(device)
    idx_test = torch.LongTensor(idx[n_train + n_val:]).to(device)

    model = GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

        print(f'Epoch {epoch+1:03d}: '
              f'Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, '
              f'Val Loss: {loss_val:.4f}, Val Acc: {acc_val:.4f}')

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter == patience:
            print("Early stopping!")
            break

    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        y_true = labels[idx_test].cpu().numpy()
        y_pred = output[idx_test].cpu().numpy().argmax(axis=1)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print("\nTest set results:")
    print(f"  Loss      = {loss_test:.4f}")
    print(f"  Accuracy  = {acc:.4f}")
    print(f"  F1 score  = {f1:.4f}")
    print(f"  Precision = {precision:.4f}")
    print(f"  Recall    = {recall:.4f}")

if __name__ == '__main__':
    try:
        import sklearn
    except ImportError:
        import sys
        !{sys.executable} -m pip install scikit-learn
    main()


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_custom_citeseer(node_path, link_path):
    node_ids = []
    features = []
    labels = []
    with open(node_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_ids.append(parts[0])
            features.append([float(x) for x in parts[1:-1]])
            labels.append(parts[-1])
    node_ids = np.array(node_ids)
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)
    classes = {c: i for i, c in enumerate(sorted(set(labels)))}
    labels = np.array([classes[x] for x in labels], dtype=np.int64)
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    edges = []
    with open(link_path, 'r') as f:
        for line in f:
            src, dst = line.strip().split()
            if src in id2idx and dst in id2idx:
                edges.append((id2idx[src], id2idx[dst]))
    edges = np.array(edges, dtype=np.int32)
    N = len(node_ids)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
                        shape=(N, N), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return features, labels, adj

def normalize_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.spmm(adj, x)
        return self.linear(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid, dropout)
        self.gc2 = GCNLayer(nhid, nclass, dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

def run_experiment(features, labels, adj, nhid, dropout, lr, weight_decay, device):
    n = features.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.2 * n)
    n_val = int(0.2 * n)
    idx_train = torch.LongTensor(idx[:n_train]).to(device)
    idx_val = torch.LongTensor(idx[n_train:n_train + n_val]).to(device)
    idx_test = torch.LongTensor(idx[n_train + n_val:]).to(device)

    model = GCN(nfeat=features.shape[1], nhid=nhid, nclass=labels.max().item() + 1, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter == patience:
            break

    # Test and metrics
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        y_true = labels[idx_test].cpu().numpy()
        y_pred = output[idx_test].cpu().numpy().argmax(axis=1)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Params: nhid={nhid}, dropout={dropout}, lr={lr}, weight_decay={weight_decay}")
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}\n")
    return {
        "loss": loss_test.item(),
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "nhid": nhid,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay
    }

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    node_path = '/content/citeseer/node'
    link_path = '/content/citeseer/link'

    features, labels, adj = load_custom_citeseer(node_path, link_path)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)

    # Only three parameter combinations
    param_grid = [
        # (nhid, dropout, lr, weight_decay)
        (16, 0.2, 0.01, 5e-4),
        (32, 0.3, 0.005, 1e-4),
        (64, 0.5, 0.01, 1e-4)
    ]

    print("Running 3 hyperparameter combinations...\n")
    results = []
    for nhid, dropout, lr, weight_decay in param_grid:
        res = run_experiment(features, labels, adj, nhid, dropout, lr, weight_decay, device)
        results.append(res)

    # Print summary table
    print("\nSummary of Results:")
    print(" nhid | dropout |   lr   | weight_decay | Accuracy |   F1    | Precision | Recall")
    print("-"*79)
    for res in results:
        print(f" {res['nhid']:>4} |  {res['dropout']:.2f}   | {res['lr']:.4f} |   {res['weight_decay']:.0e}   |  {res['accuracy']:.4f} | {res['f1']:.4f} |  {res['precision']:.4f} | {res['recall']:.4f}")

if __name__ == '__main__':
    # If in Colab, install scikit-learn if needed
    try:
        import sklearn
    except ImportError:
        import sys
        !{sys.executable} -m pip install scikit-learn
    main()
