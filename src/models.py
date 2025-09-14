import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallerCNN(nn.Module):
    """A smaller CNN model for MNIST classification."""
    def __init__(self):
        super(SmallerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ArrhythmiaMLP(nn.Module):
    """
    A deeper and wider DP-compatible MLP for the Arrhythmia dataset.
    Uses GroupNorm and LeakyReLU.
    """
    def __init__(self, num_features, num_classes):
        super(ArrhythmiaMLP, self).__init__()
        self.layer_1 = nn.Linear(num_features, 512)
        self.groupnorm1 = nn.GroupNorm(32, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.groupnorm2 = nn.GroupNorm(32, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.groupnorm3 = nn.GroupNorm(32, 128)
        self.layer_out = nn.Linear(128, num_classes)
        
        self.act = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(self.act(self.groupnorm1(self.layer_1(x))))
        x = self.dropout(self.act(self.groupnorm2(self.layer_2(x))))
        x = self.dropout(self.act(self.groupnorm3(self.layer_3(x))))
        x = self.layer_out(x)
        return x

class BatteryLSTM(nn.Module):
    """
    An LSTM-based model to predict Remaining Useful Life (RUL) of batteries.
    This is a regression model.
    """
    # --- ADD drop_prob to the constructor ---
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim=1, drop_prob=0.2):
        super(BatteryLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Use the passed drop_prob
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def get_model(config):
    """Factory function to return the appropriate model."""
    model_name = config.get('model_name')
    if model_name == 'cnn':
        return SmallerCNN()
    elif model_name == 'mlp':
        return ArrhythmiaMLP(
            num_features=config['num_features'], 
            num_classes=config['num_classes']
        )
    elif model_name == 'lstm_battery':
        return BatteryLSTM(
            input_dim=config['num_features'],
            hidden_dim=config['lstm_hidden_dim'],
            n_layers=config['lstm_n_layers'],
            drop_prob=config.get('lstm_drop_prob', 0.2) # Get from config, default to 0.2
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")