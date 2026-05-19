import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载CSV
df = pd.read_csv('AAPL_stock_data.csv', index_col=0)

# 强制将Close列转换为数值类型
close_series = pd.to_numeric(df['Close'], errors='coerce').dropna()

# 将pandas数据转换为标准的numpy数组
data = close_series.to_numpy().reshape(-1, 1)

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 构造滑动窗口
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

SEQ_LENGTH = 30
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# 转换为PyTorch张量
# 检查GPU（RTX 4060）是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

X_train = torch.from_numpy(X).float().to(device)
y_train = torch.from_numpy(y).float().to(device)

print(f"数据处理完成！训练集形状: {X_train.shape}")

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        # input_size: 输入特征数（只用收盘价，设置为1）
        # hidden_size: 隐藏层神经元数量
        # num_layers: LSTM堆叠的层数
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态h0和细胞状态c0，送入GPU
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


# 实例化模型并送入GPU
model = StockLSTM().to(device)
print(model)

criterion = nn.MSELoss()  # 均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 学习率设为0.001

num_epochs = 100  # 训练轮数
loss_history = []  # 用于记录每轮的损失

print("开始训练模型...")
for epoch in range(num_epochs):
    model.train()

    # 计算预测值
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 更新权重
    optimizer.zero_grad()  # 清除之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("训练完成！")

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='训练损失 (Training Loss)', color='orange')
plt.title('模型训练损失下降曲线')
plt.xlabel('迭代轮数 (Epoch)')
plt.ylabel('损失值 (Loss)')
plt.legend()
plt.grid(True)
plt.show()

# 模型预测与数据还原
model.eval()

with torch.no_grad():
    predictions = model(X_train).cpu().numpy()  # 将结果从GPU拿回CPU绘图

# 还原真实价格
y_train_real = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
predictions_real = scaler.inverse_transform(predictions)

# 计算评估指标：均方根误差 (RMSE)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_train_real, predictions_real))
print(f"模型的均方根误差 (RMSE): {rmse:.2f}")

# 可视化预测结果
plt.figure(figsize=(14, 7))
plt.plot(y_train_real, label='真实收盘价 (Actual)', color='blue', alpha=0.7)
plt.plot(predictions_real, label='模型预测价 (Predicted)', color='red', linestyle='--')
plt.title(f'股票价格预测对比图 (RMSE: {rmse:.2f})')
plt.xlabel('交易日天数')
plt.ylabel('价格 (USD)')
plt.legend()
plt.grid(True)
plt.show()
