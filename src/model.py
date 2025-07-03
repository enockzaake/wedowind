import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

print("Loading csv...")
# 1. Load and visualize data
df = pd.read_csv("datasets/rotor_icing.csv")
df = df.sort_values("timestamp")

# print(df.head())


# Visualize features over time
sns.lineplot(x='timestamp', y='wind_speed', data=df)
plt.title("Wind Speed Over Time")
plt.xticks(rotation=45)
plt.show()

# 2. Feature Engineering: Extract rolling and derived features
df["wind_power_ratio"] = df["power_output"] / (df["wind_speed"]**3 + 1e-3)
df["rotor_acceleration"] = df["rotor_speed"].diff().fillna(0)
df["temp_diff"] = df["temperature"].diff().fillna(0)

# Rolling mean features
for col in ["wind_speed", "power_output", "rotor_speed", "temperature"]:
    df[f"{col}_rolling_mean"] = df[col].rolling(window=10, min_periods=1).mean()
    df[f"{col}_rolling_std"] = df[col].rolling(window=10, min_periods=1).std().fillna(0)

features = [
    "wind_speed", "power_output", "rotor_speed", "temperature",
    "wind_power_ratio", "rotor_acceleration", "temp_diff",
    "wind_speed_rolling_mean", "wind_speed_rolling_std",
    "power_output_rolling_mean", "power_output_rolling_std",
    "rotor_speed_rolling_mean", "rotor_speed_rolling_std",
    "temperature_rolling_mean", "temperature_rolling_std"
]

# 3. Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
y = df["icing"].values.astype(np.float32)

# 4. Create sequences for LSTM input
def create_sequences(data, labels, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        targets.append(labels[i + window_size])
    return np.array(sequences), np.array(targets)

SEQ_LEN = 20
X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64)

# 5. Define LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        decoded, _ = self.decoder(h.repeat(x.size(1), 1, 1).permute(1, 0, 2))
        return decoded

# 6. Train LSTM Autoencoder
autoencoder = LSTMAutoencoder(input_dim=X_train.shape[2], hidden_dim=32)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    autoencoder.train()
    losses = []
    for xb, _ in train_loader:
        optimizer.zero_grad()
        out = autoencoder(xb)
        loss = criterion(out, xb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")

# 7. Detect anomalies using reconstruction error
autoencoder.eval()
with torch.no_grad():
    reconstructions = autoencoder(X_test_t)
    errors = torch.mean((reconstructions - X_test_t) ** 2, dim=(1, 2))

threshold = errors.mean() + errors.std()
predictions_ae = (errors > threshold).float()

print("\nLSTM Autoencoder Evaluation:")
print(confusion_matrix(y_test_t, predictions_ae))
print(classification_report(y_test_t, predictions_ae))

# 8. CNN + LSTM Model
class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # batch, features, seq_len
        x = self.cnn(x)  # batch, channels, seq_len/2
        x = x.permute(0, 2, 1)  # batch, seq_len/2, channels
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h[-1]))

model = CNNLSTMClassifier(input_dim=X_train.shape[2], hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# 9. Train CNN+LSTM
for epoch in range(10):
    model.train()
    losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"CNN+LSTM Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")

# 10. Evaluate CNN+LSTM
model.eval()
with torch.no_grad():
    preds = model(X_test_t).squeeze()
    preds_cls = (preds > 0.5).float()

print("\nCNN+LSTM Evaluation:")
print(confusion_matrix(y_test_t, preds_cls))
print(classification_report(y_test_t, preds_cls))

# 11. Visualization of reconstruction error
plt.figure(figsize=(10, 5))
plt.plot(errors.numpy(), label="Reconstruction Error")
plt.axhline(threshold.item(), color="red", linestyle="--", label="Threshold")
plt.title("LSTM Autoencoder Reconstruction Error")
plt.legend()
plt.show()
