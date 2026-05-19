import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 加载第一阶段保存的数据
    df = pd.read_csv('processed_chatgpt_data.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')

    # 定义扩散强度
    df['Intensity'] = (df['RetweetCount'] + df['LikeCount'] + 1) * df['Sentiment_Score'].abs()

    # 按时间重采样
    resampled_df = df.resample('6h', on='Datetime').agg({
        'Intensity': 'sum',
        'Text': 'count'
    }).rename(columns={'Text': 'Tweet_Volume'})

    # 可视化
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制扩散覆盖范围
    ax1.set_xlabel('Time (Jan 2023)')
    ax1.set_ylabel('Diffusion Reach (Tweet Count)', color='tab:blue')
    ax1.fill_between(resampled_df.index, resampled_df['Tweet_Volume'], color='tab:blue', alpha=0.3)
    ax1.plot(resampled_df.index, resampled_df['Tweet_Volume'], color='tab:blue', linewidth=2, label='Reach')

    # 绘制扩散强度
    ax2 = ax1.twinx()
    ax2.set_ylabel('Infection Intensity (Influence Score)', color='tab:red')
    ax2.plot(resampled_df.index, resampled_df['Intensity'], color='tab:red', linestyle='--', linewidth=2,
             label='Intensity')

    plt.title('ChatGPT Social Network Diffusion Simulation', fontsize=16)
    fig.tight_layout()

    # 保存结果
    plt.savefig('diffusion_simulation.png')
    print("扩散模拟图已保存为: diffusion_simulation.png")
    plt.show()


if __name__ == "__main__":
    main()