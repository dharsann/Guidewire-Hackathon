import torch
import torch.nn as nn
import joblib

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

input_size = 3
hidden_size = 50
num_layers = 2
output_size = 3

prediction_model = LSTMModel(input_size, hidden_size, num_layers, output_size)

prediction_model.load_state_dict(torch.load("models/lstm_k8s_model.pth"))
prediction_model.eval()

anomaly_detection_model = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_kubernetes_metrics(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        predicted_values = prediction_model(input_tensor).numpy().tolist()
    return predicted_values

def detect_anomaly(predicted_values):
    predicted_values_scaled = scaler.transform([predicted_values[0]]) 
    prediction = anomaly_detection_model.predict(predicted_values_scaled)
    return "Anomaly" if prediction[0] == -1 else "Normal"