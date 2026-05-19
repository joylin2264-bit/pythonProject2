# 基于大数据框架的金融股息数据分析与预测系统

## 项目概述
本课设旨在利用大数据技术（Hadoop/Hive/Spark）构建金融时序数据的处理流水线，结合深度学习模型（LSTM/GRU）对 AAPL 股票数据进行趋势预测与分析。

## 技术栈
- **存储层:** Apache Hive (基于 Docker)
- **计算层:** Apache Spark (本地分布式并行算子)
- **模型层:** PyTorch (LSTM/GRU 深度学习)
- **工具:** PyCharm, Ubuntu (WSL/VMware), GitHub

## 目录结构
```text
/
├── data/                  # 存放原始 CSV 及预处理后的特征矩阵
├── scripts/               # 核心源码
│   ├── data_ingest.sql    # Hive 建表与入库脚本
│   ├── spark_process.py   # Spark 特征工程脚本
│   └── train_predict.py   # PyTorch 模型训练与可视化脚本
├── README.md              # 项目文档
└── results/           # 实验结果分析图表
