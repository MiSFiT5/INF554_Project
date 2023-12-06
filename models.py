import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """LSTM-based classifier."""
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass of the LSTM model."""
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        output = self.sigmoid(output)
        return output

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Graph Convolutional Network (GCN) model."""
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        """Forward pass of the GCN model."""
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1, dropout=0.3):
        """Graph Attention Network (GAT) model."""
        super(GATModel, self).__init__()
        self.convs = nn.ModuleList([
            GATConv(input_dim, hidden_dim, heads=num_heads),
            GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
        ])
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * num_heads, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        """Forward pass of the GAT model."""
        x, edge_index = data.x, data.edge_index
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = self.fc(x)
        return x

class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        """GraphSAGE model with attention mechanism."""
        super(GraphSAGEModel, self).__init__()
        self.convs = nn.ModuleList([
            SAGEConv(input_dim, hidden_dim*4),
            SAGEConv(hidden_dim*4, hidden_dim*2),
            SAGEConv(hidden_dim*2, hidden_dim),
        ])

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        """Forward pass of the GraphSAGE model."""
        x, edge_index = data.x, data.edge_index

        # Apply multiple GraphSAGE layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)

        # Attention mechanism
        attention_weights = F.softmax(self.attention(x), dim=0)
        x = x * attention_weights

        # Fully connected layer for final prediction
        x = self.fc(x)

        return x
