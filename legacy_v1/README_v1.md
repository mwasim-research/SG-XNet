
# SG-XNet: Structured Graph eXtended Network

This repository contains the implementation of **SG-XNet (Structured Graph eXtended Network)**, a hybrid Graph Neural Network (GNN) model for transfer learning in node classification tasks.

SG-XNet combines the inductive learning capability of **GraphSAGE** with the structured feature propagation of **Graph Convolutional Networks (GCN)**. The repository also includes baseline implementations of GCN, GraphSAGE, Graph Isomorphism Network (GIN), and Graph Attention Network (GAT) for comparison.

## Overview

Graph Neural Networks achieve strong performance on node classification tasks but often struggle to transfer across graphs with different structures and label distributions. SG-XNet is designed to improve transfer learning by combining:

- GraphSAGE for inductive neighborhood aggregation
- GCN layers for structured feature propagation
- Dropout regularization for improved generalization

## Repository Structure

```text
Fine-tuning/
├── GAT/
│   └── gat_finetuning.py
├── GCN/
│   └── gcn_finetuning.py
├── GIN/
│   └── gin_finetuning.py
├── GraphSAGE/
│   └── graphsage_finetuning.py
└── SG-XNet/
    └── sgxnet_finetuning.py

SGXNet.png
README.md
requirements.txt
