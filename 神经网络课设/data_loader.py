import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 使用代理
proxy = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义要获取的股票代码和时间范围
stock_symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2023-12-31'

print(f"正在从 Yahoo Finance 下载 {stock_symbol} 的历史交易数据...")

# 使用 yfinance 下载真实数据
df = yf.download(stock_symbol, start=start_date, end=end_date)

# 验证数据是否获取成功
if df.empty:
    print("数据下载失败！")
else:
    print("数据下载成功！前 5 条数据如下：")
    print(df.head())

    # 保存为 CSV 文件
    file_name = f"{stock_symbol}_stock_data.csv"
    df.to_csv(file_name)
    print(f"\n数据已成功保存至当前目录：{file_name}")

    # 绘制收盘价走势图
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='每日收盘价 (Close Price)', color='blue')
    plt.title(f"{stock_symbol} 历史收盘价真实走势图 (2015-2023)")
    plt.xlabel("日期 (Date)")
    plt.ylabel("价格 (USD)")
    plt.legend()
    plt.grid(True)

    print("\n正在生成可视化图表...")
    plt.show()
