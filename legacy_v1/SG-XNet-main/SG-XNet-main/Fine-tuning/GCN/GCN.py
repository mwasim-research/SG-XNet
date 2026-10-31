#@title GCN 64 channels 5 layers
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)  # Dropout after each hidden layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)  # No dropout after output layer

def train(model, optimizer, data, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

        pred = out.argmax(dim=1)
        correct = (pred == data.y).sum().item()
        acc = correct / len(data.y)

        y_true = data.y.cpu().numpy()
        y_pred = pred.cpu().numpy()

        precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

        # Calculate log loss
        y_prob = out.exp().cpu().numpy()  # Convert log probabilities back to probabilities
        logloss = log_loss(y_true, y_prob)

    return acc, precision, recall, f1, logloss

def pretrain_citeseer(model, optimizer, data, epochs=200):
    criterion = torch.nn.NLLLoss()

    best_acc = 0.0
    total_acc, total_precision, total_recall, total_f1, total_logloss = 0.0, 0.0, 0.0, 0.0, 0.0  # Accumulate metrics

    for epoch in tqdm(range(epochs)):
        loss = train(model, optimizer, data, criterion)
        acc, precision, recall, f1, logloss = evaluate(model, data)

        # Accumulate metrics over epochs
        total_acc += acc
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_logloss += logloss

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'gcn_citeseer.pth')

    # Calculate averages
    avg_acc = total_acc / epochs
    avg_precision = total_precision / epochs
    avg_recall = total_recall / epochs
    avg_f1 = total_f1 / epochs
    avg_logloss = total_logloss / epochs

    return best_acc, avg_acc, avg_precision, avg_recall, avg_f1, avg_logloss

def initialize_model_for_cora(in_channels, out_channels):
    new_model = GCN(in_channels=in_channels, hidden_channels=64, out_channels=out_channels, num_layers=5).to(device)
    return new_model

def load_pretrained_weights_for_cora(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()

    # Filter out keys that are not in the current model
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Match sizes of the parameters
    for k, v in filtered_dict.items():
        if model_dict[k].shape != v.shape:
            print(f"Skipping loading parameter {k} due to size mismatch.")
            continue
        model_dict[k] = v

    model.load_state_dict(model_dict, strict=False)

def finetune_cora(model, data, epochs=200):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    best_acc = 0.0
    total_acc, total_precision, total_recall, total_f1, total_logloss = 0.0, 0.0, 0.0, 0.0, 0.0  # Accumulate metrics

    for epoch in tqdm(range(epochs)):
        loss = train(model, optimizer, data, criterion)
        acc, precision, recall, f1, logloss = evaluate(model, data)

        # Accumulate metrics over epochs
        total_acc += acc
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_logloss += logloss

        # Print metrics for each epoch
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, LogLoss: {logloss:.4f}')

        if acc > best_acc:
            best_acc = acc

    # Calculate averages
    avg_acc = total_acc / epochs
    avg_precision = total_precision / epochs
    avg_recall = total_recall / epochs
    avg_f1 = total_f1 / epochs
    avg_logloss = total_logloss / epochs

    return best_acc, avg_acc, avg_precision, avg_recall, avg_f1, avg_logloss

# Pretraining on Citeseer
citeseer_dataset = Planetoid(root='data/Planetoid', name='Citeseer')
citeseer_data = citeseer_dataset[0].to(device)

# Initialize the model for Citeseer
model_citeseer = initialize_model_for_cora(
    in_channels=citeseer_data.num_features,
    out_channels=citeseer_dataset.num_classes
)

# Optimizer for pretraining on Citeseer
optimizer_citeseer = torch.optim.Adam(model_citeseer.parameters(), lr=0.01, weight_decay=0.0005)

# Pretrain the model on Citeseer
print("Pretraining on Citeseer")
best_pretrained_acc, avg_pretrained_acc, avg_precision, avg_recall, avg_f1, avg_logloss = pretrain_citeseer(model_citeseer, optimizer_citeseer, citeseer_data)
print(f'Best Pretrained Accuracy on Citeseer: {best_pretrained_acc:.4f}, Average Accuracy: {avg_pretrained_acc:.4f}, Avg Precision: {avg_precision:.4f}, Avg Recall: {avg_recall:.4f}, Avg F1: {avg_f1:.4f}, Avg LogLoss: {avg_logloss:.4f}')

# Load Cora dataset
cora_dataset = Planetoid(root='data/Planetoid', name='Cora')
cora_data = cora_dataset[0].to(device)

# Initialize a new model for Cora with correct dimensions
model_cora = initialize_model_for_cora(
    in_channels=cora_data.num_features,
    out_channels=cora_dataset.num_classes
)

# Load the pretrained weights (for the common layers only)
load_pretrained_weights_for_cora(model_cora, 'gcn_citeseer.pth')

# Finetune on Cora
print("Finetuning on Cora")
best_finetuned_acc, avg_finetuned_acc, avg_precision, avg_recall, avg_f1, avg_logloss = finetune_cora(model_cora, cora_data)
print(f'Finetuned Accuracy on Cora: {best_finetuned_acc:.4f}, Average Accuracy: {avg_finetuned_acc:.4f}, Avg Precision: {avg_precision:.4f}, Avg Recall: {avg_recall:.4f}, Avg F1: {avg_f1:.4f}, Avg LogLoss: {avg_logloss:.4f}')
