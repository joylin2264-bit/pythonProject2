import yfinance as yf
import os

# 使用代理
proxy = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy

# 从雅虎下载苹果公司的股票数据，时间跨度从2015年到2024年
ticker = "AAPL"
print(f"正在下载 {ticker} 的历史数据...")

stock = yf.Ticker(ticker)
df = stock.history(start="2015-01-01", end="2024-01-01")

# 调整重置索引
df = df.reset_index()

# 只保留核心字段：日期、收盘价、最高价、最低价、开盘价、成交量、股息
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']]

# 格式化日期为 YYYY-MM-DD
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

# 自动保存到 data 文件夹下
output_dir = "../data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "AAPL_stock_dividends.csv")
df.to_csv(output_path, index=False)

print(f"数据集下载成功，文件已保存至: {output_path}")
print("数据前5行预览：")
print(df.head())
