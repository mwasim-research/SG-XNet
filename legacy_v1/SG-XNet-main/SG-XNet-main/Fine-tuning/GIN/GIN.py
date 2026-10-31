#@title GIN 64 channels 5 layers
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the GIN model with 5 layers
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5):  # num_layers set to 5
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()

        # First layer
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn1))

        # Hidden layers
        for _ in range(num_layers - 2):  # Adding hidden layers
            nn_hidden = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn_hidden))

        # Output layer
        nn_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.convs.append(GINConv(nn_out))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)  # Log softmax for stable log loss calculation

# Function to train the model
def train(model, optimizer, data, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Function to evaluate the model
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)  # Log probabilities
        pred = out.argmax(dim=1)  # Predicted class labels
        correct = (pred == data.y).sum().item()
        acc = correct / len(data.y)

        y_true = data.y.cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = torch.exp(out).cpu().numpy()  # Convert log probabilities to regular probabilities

        precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
        logloss_value = log_loss(y_true, y_prob)

    return acc, precision, recall, f1, logloss_value

# Pretraining on Citeseer
def pretrain_citeseer(model, optimizer, data, epochs=200):
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in tqdm(range(epochs)):
        loss = train(model, optimizer, data, criterion)
        acc = evaluate(model, data)[0]
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'gin_citeseer.pth')  # Save GIN model
    return best_acc

# Function to initialize the model with 5 layers
def initialize_model(in_channels, out_channels, hidden_channels=64, num_layers=5):  # Set num_layers to 5
    model = GIN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers).to(device)
    return model

# Function to load pretrained weights
def load_pretrained_weights(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()

    # Filter out keys that are not in the current model
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Track loaded layers
    loaded_layers = []
    skipped_layers = []

    # Match sizes of the parameters
    for k, v in filtered_dict.items():
        if model_dict[k].shape != v.shape:
            skipped_layers.append(k)
            print(f"Skipping loading parameter {k} due to size mismatch.")
            continue
        model_dict[k] = v
        loaded_layers.append(k)

    model.load_state_dict(model_dict, strict=False)

    # Print information about loaded and skipped layers
    print(f"Loaded layers: {loaded_layers}")
    print(f"Skipped layers: {skipped_layers}")

# Function to fine-tune on the Cora dataset and calculate average metrics
def finetune_cora(model, data, epochs=200):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    best_acc = 0.0
    total_acc, total_precision, total_recall, total_f1, total_logloss = 0, 0, 0, 0, 0

    for epoch in tqdm(range(epochs)):
        loss = train(model, optimizer, data, criterion)
        acc, precision, recall, f1, logloss_value = evaluate(model, data)

        total_acc += acc
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_logloss += logloss_value

        if acc > best_acc:
            best_acc = acc

    avg_acc = total_acc / epochs
    avg_precision = total_precision / epochs
    avg_recall = total_recall / epochs
    avg_f1 = total_f1 / epochs
    avg_logloss = total_logloss / epochs

    # Print average metrics at the end of training
    print(f'Final Metrics - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Log Loss: {logloss_value:.4f}')
    print(f'Average Metrics over {epochs} epochs - Accuracy: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}, Log Loss: {avg_logloss:.4f}')

    return best_acc, avg_acc

# Pretraining on Citeseer
citeseer_dataset = Planetoid(root='data/Planetoid', name='Citeseer')
citeseer_data = citeseer_dataset[0].to(device)

# Initialize the model for Citeseer with 5 layers
model_citeseer = initialize_model(
    in_channels=citeseer_data.num_features,
    out_channels=citeseer_dataset.num_classes,
    num_layers=5  # Set to 5 layers
)

# Optimizer for pretraining on Citeseer
optimizer_citeseer = torch.optim.Adam(model_citeseer.parameters(), lr=0.01, weight_decay=0.0005)

# Pretrain the model on Citeseer
print("Pretraining on Citeseer")
best_pretrained_acc = pretrain_citeseer(model_citeseer, optimizer_citeseer, citeseer_data)
print(f'Best Pretrained Accuracy on Citeseer: {best_pretrained_acc:.4f}')

# Load Cora dataset
cora_dataset = Planetoid(root='data/Planetoid', name='Cora')
cora_data = cora_dataset[0].to(device)

# Initialize a new model for Cora with correct dimensions and 5 layers
model_cora = initialize_model(
    in_channels=cora_data.num_features,
    out_channels=cora_dataset.num_classes,
    num_layers=5  # Set to 5 layers
)

# Load the pretrained weights (for the common layers only)
load_pretrained_weights(model_cora, 'gin_citeseer.pth')  # Load GIN model

# Finetune on Cora (fine-tune all layers, including pretrained ones)
print("Finetuning on Cora")
best_finetuned_acc, avg_finetuned_acc = finetune_cora(model_cora, cora_data)
print(f'Best Finetuned Accuracy on Cora: {best_finetuned_acc:.4f}')
print(f'Average Finetuned Accuracy on Cora: {avg_finetuned_acc:.4f}')
