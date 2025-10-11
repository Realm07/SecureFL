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

class LSTMAttention(nn.Module):
    """
    An LSTM model with a self-attention mechanism, inspired by recent research
    for high-accuracy RUL prediction.
    """
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim=1, drop_prob=0.2):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob, bidirectional=True)
        
        # Attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # *2 because bidirectional
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        attention_weights = self.attention_layer(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.fc(context_vector)
        return out
    
class SimpleMLP(nn.Module):
    """A simple MLP model for MNIST, includes a flatten layer."""
    def __init__(self, num_features=784, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # IMPORTANT: Flatten the image from [batch_size, 1, 28, 28] to [batch_size, 784]
        x = x.view(x.shape[0], -1) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(config):
    """Factory function to return the appropriate model."""
    model_name = config.get('model_name')
    
    # --- NEW LOGIC HERE ---
    # If the model is mlp AND the dataset is mnist, use our new SimpleMLP
    if model_name == 'mlp' and config.get('dataset_name') == 'mnist':
        return SimpleMLP(
            num_features=config['num_features'],
            num_classes=config.get('num_classes', 10) # Default to 10 for MNIST
        )
    
    if model_name == 'cnn':
        return SmallerCNN()
    elif model_name == 'mlp': # This will now only catch Arrhythmia
        return ArrhythmiaMLP(
            num_features=config['num_features'], 
            num_classes=config['num_classes']
        )
    elif model_name == 'lstm_battery':
        return BatteryLSTM(
            input_dim=config['num_features'],
            hidden_dim=config['lstm_hidden_dim'],
            n_layers=config['lstm_n_layers'],
            drop_prob=config.get('lstm_drop_prob', 0.2)
        )
    elif model_name == 'lstm_attention':
        return LSTMAttention(
            input_dim=config['num_features'],
            hidden_dim=config['lstm_hidden_dim'],
            n_layers=config['lstm_n_layers'],
            drop_prob=config.get('lstm_drop_prob', 0.2)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")