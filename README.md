# Graph Convolutional Neural Networks

Unofficial Reimplementation of "Semi-Supervised Classification with Graph Convolutional Networks"[^1][^2] in PyTorch.

## Usage

### Dense

```python
input = ...
adj = ...

n_nodes = adj.size(0)  # or n_nodes = input.size(1)
n_features = input.size(-1)
h1_features = ...
h2_features = ...

conv1 = nn.Sequential(LinearGraphConv(n_features, h1_features), nn.ReLU())
conv2 = nn.Sequential(LinearGraphConv(h1_features, h2_features), nn.ReLU())

output = conv2((conv1((input, adj)), adj))
```

### Sparse

```python
input = ...
adj_sparse_coo = ...  

n_nodes = adj.size(0)  # or n_nodes = input.size(1)
n_features = input.size(-1)
h1_features = ...
h2_features = ...

conv1 = nn.Sequential(SparseLinearGraphConv(n_features, h1_features), nn.ReLU())
conv2 = nn.Sequential(SparseLinearGraphConv(h1_features, h2_features), nn.ReLU())

output = conv2((conv1((input, adj_sparse_coo)), adj_sparse_coo))
```

## TODOs

Add Cora dataset example.

## References

[^1]: [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907).
[^2]: [Official Implementation](https://github.com/tkipf/gcn).
