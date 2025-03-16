import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("data/kubernetes_performance_metrics_dataset.csv")

features = ["cpu_allocation_efficiency", "memory_allocation_efficiency", "disk_io"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

joblib.dump(scaler, "scaler.pkl")

SEQ_LENGTH = 10

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

data = df[features].values
X, y = create_sequences(data, SEQ_LENGTH)

X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    
input_size = len(features)
hidden_size = 50
num_layers = 2
output_size = len(features)

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "lstm_k8s_model.pth")
print("Detection model saved successfully!")

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df_scaled = scaler.fit_transform(df[features])
model.fit(df_scaled)

joblib.dump(model, "isolation_forest.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Anomaly detection model trained and saved!")