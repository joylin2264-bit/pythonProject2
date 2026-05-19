# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查硬件配置（自动调用 RTX 4060 GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前硬件设备: {device}")

# 读取前面 Spark 算好的高维特征数据
data_path = "../data/AAPL_processed_features.csv"
df = pd.read_csv(data_path)

# 选择训练特征与预测目标（以收盘价及股息衍生特征进行联合预测）
features = df[['close_price', 'dividend_yield', 'close_ma5']].values
target = df['close_price'].values.reshape(-1, 1)

# 数据标准化（将特征缩放到 0~1 之间，加速神经网络收敛）
scaler_feat = MinMaxScaler()
scaler_targ = MinMaxScaler()
features_scaled = scaler_feat.fit_transform(features)
target_scaled = scaler_targ.fit_transform(target)

# 构造滑动时间窗口（用过去 10 天的特征预测第 11 天的股价）
def create_sequences(features, target, seq_length=10):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(features_scaled, target_scaled)

# 划分训练集与测试集（80%数据用于训练，20%用于验证模型精度）
split = int(len(X) * 0.8)
X_train, X_test = torch.tensor(X[:split], dtype=torch.float32).to(device), torch.tensor(X[split:], dtype=torch.float32).to(device)
y_train, y_test = torch.tensor(y[:split], dtype=torch.float32).to(device), torch.tensor(y[split:], dtype=torch.float32).to(device)

# 定义全局网络超参数
input_size = 3    # 3个输入特征
hidden_size = 64  # 隐藏层神经元数量
num_layers = 2    # 2层网络叠加
output_size = 1   # 预测1个未来价格

# 定义 LSTM 模型架构
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 定义 GRU 模型架构
class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# 训练函数
def train_model(model, epochs=30):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

print("\n开始并行训练 LSTM 深度学习网络...")
lstm_model = train_model(LSTMModel().to(device))
print("开始并行训练 GRU 深度学习网络...")
gru_model = train_model(GRUModel().to(device))

# 模型预测与逆标准化还原
lstm_model.eval()
gru_model.eval()
with torch.no_grad():
    lstm_preds = lstm_model(X_test).cpu().numpy()
    gru_preds = gru_model(X_test).cpu().numpy()

# 将 0~1 的数据还原为真实的股票价格美元单位
actual_prices = scaler_targ.inverse_transform(y_test.cpu().numpy())
lstm_decoded = scaler_targ.inverse_transform(lstm_preds)
gru_decoded = scaler_targ.inverse_transform(gru_preds)

# 计算核心评估指标：均方根误差 (RMSE)
lstm_rmse = np.sqrt(np.mean((actual_prices - lstm_decoded) ** 2))
gru_rmse = np.sqrt(np.mean((actual_prices - gru_decoded) ** 2))
print(f"\n实验评估结果：")
print(f"LSTM 模型的预测 RMSE 指标: {lstm_rmse:.4f}")
print(f"GRU 模型的预测 RMSE 指标:  {gru_rmse:.4f}")

# 绘制高阶数据可视化曲线
print("\n正在生成多模型预测对比可视化图表...")
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label="真实市场价格 (Actual Price)", color='black', linewidth=1.5)
plt.plot(lstm_decoded, label=f"LSTM 预测曲线 (RMSE: {lstm_rmse:.2f})", color='crimson', linestyle='--')
plt.plot(gru_decoded, label=f"GRU 预测曲线 (RMSE: {gru_rmse:.2f})", color='royalblue', linestyle=':')
plt.title("基于大数据特征提炼下的 AAPL 股价与股息趋势预测对比", fontsize=14)
plt.xlabel("测试集时间跨度 (天)", fontsize=12)
plt.ylabel("价格 (USD)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# 自动保存图表
output_img = "../data/prediction_comparison.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"可视化结果图表已保存至: {output_img}")
plt.show()
