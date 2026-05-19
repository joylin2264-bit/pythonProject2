# -*- coding: utf-8 -*-
"""
大数据编程课程设计：基于 Spark 的股息数据预处理与特征工程（Windows本地运行版）
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg
from pyspark.sql.window import Window
import os


def main():

    # 1. 初始化本地 Spark 会话（利用 Windows 本地多核并行计算）
    spark = SparkSession.builder \
        .appName("StockDividendFeatureEngineeringLocal") \
        .master("local[*]") \
        .getOrCreate()

    # 2. 绕过 Docker，直接高效读取本地的真实股息数据集
    input_path = "../data/AAPL_stock_dividends.csv"
    print(f"正在通过 Spark 分布式算子读取本地数据集: {input_path}")

    # 自动识别 CSV 表头和数据类型
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # 3. 特征工程 A：计算“股息率” (Dividend Yield)
    print("正在利用 Spark 分布式并行计算：每日股息率 (Dividend Yield)...")
    df_with_yield = df.withColumn(
        "dividend_yield",
        when(col("Dividends") > 0, col("Dividends") / col("Close")).otherwise(0.0)
    )

    # 4. 特征工程 B：计算“5日移动平均收盘价” (5-day Moving Average)
    print("正在利用 Spark 窗口函数计算：5日移动平均收盘价...")
    window_spec = Window.orderBy("Date").rowsBetween(-4, 0)
    df_with_ma = df_with_yield.withColumn("close_ma5", avg(col("Close")).over(window_spec))

    # 5. 特征工程 C：构建“股息发放指示器” (Is_Dividend_Day)
    print("正在构建分类特征：股息发放指示器 (Is_Dividend_Day)...")
    df_final = df_with_ma.withColumn(
        "is_dividend_day",
        when(col("Dividends") > 0, 1).otherwise(0)
    )

    # 重命名列名以符合规范
    df_final = df_final.withColumnRenamed("Date", "event_date") \
        .withColumnRenamed("Open", "open_price") \
        .withColumnRenamed("High", "high_price") \
        .withColumnRenamed("Low", "low_price") \
        .withColumnRenamed("Close", "close_price") \
        .withColumnRenamed("Volume", "volume") \
        .withColumnRenamed("Dividends", "dividends")

    # 6. 将加工好的分布式特征矩阵导出为本地 CSV
    print("Spark 并行特征工程计算完成，正在导出特征矩阵...")
    result_pd = df_final.toPandas()

    output_dir = "../data"
    output_path = os.path.join(output_dir, "AAPL_processed_features.csv")
    result_pd.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n特征矩阵构建成功，文件已保存至: {output_path}")
    print("加工后的高维特征数据预览（前 5 行）：")
    print(
        result_pd[['event_date', 'close_price', 'dividends', 'dividend_yield', 'close_ma5', 'is_dividend_day']].head())

    spark.stop()


if __name__ == "__main__":
    main()
