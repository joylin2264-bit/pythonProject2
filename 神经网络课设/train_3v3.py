import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# 环境与字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# 数据加载与预处理
df = pd.read_csv('AAPL_stock_data.csv', index_col=0)
close_series = pd.to_numeric(df['Close'], errors='coerce').dropna()
data = close_series.to_numpy().reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

SEQ_LENGTH = 30
X, y = create_sequences(scaled_data, SEQ_LENGTH)
X_train = torch.from_numpy(X).float().to(device)
y_train = torch.from_numpy(y).float().to(device)

# 模型定义
# 循环神经网络
class StockRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

# GRU
class StockGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

# LSTM
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(device)
        c0 = torch.zeros(2, x.size(0), 64).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# 训练函数
def train_model(model_class, name, X_train, y_train, epochs=100):
    model = model_class().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []

    print(f"\n开始训练 {name} 模型...")
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f'{name} - Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

    model.eval()
    with torch.no_grad():
        preds = model(X_train).cpu().numpy()
    return scaler.inverse_transform(preds), history

# 对比
rnn_preds, rnn_loss = train_model(StockRNN, "RNN", X_train, y_train)
lstm_preds, lstm_loss = train_model(StockLSTM, "LSTM", X_train, y_train)
gru_preds, gru_loss = train_model(StockGRU, "GRU", X_train, y_train)

# 结果可视化
# Loss 曲线对比
plt.figure(figsize=(10, 5))
plt.plot(rnn_loss, label='RNN Loss')
plt.plot(lstm_loss, label='LSTM Loss')
plt.plot(gru_loss, label='GRU Loss')
plt.title('三种模型的训练损失下降对比')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 预测结果对比
y_train_real = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))

plt.figure(figsize=(15, 8))
plt.plot(y_train_real, label='真实价格 (Actual)', color='black', linewidth=1.5)
plt.plot(rnn_preds, label='RNN 预测', alpha=0.7)
plt.plot(lstm_preds, label='LSTM 预测', alpha=0.7)
plt.plot(gru_preds, label='GRU 预测', alpha=0.7)
plt.title('RNN vs LSTM vs GRU 股票预测多模型对比')
plt.legend()
plt.show()

# 计算并打印RMSE
def print_rmse(real, pred, name):
    rmse = np.sqrt(mean_squared_error(real, pred))
    print(f"{name} 模型的均方根误差 (RMSE): {rmse:.2f}")

print("\n--- 性能评估 ---")
print_rmse(y_train_real, rnn_preds, "RNN")
print_rmse(y_train_real, lstm_preds, "LSTM")
print_rmse(y_train_real, gru_preds, "GRU")
