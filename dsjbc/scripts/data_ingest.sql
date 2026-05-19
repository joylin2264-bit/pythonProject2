
-- 创建并使用独立的数据库
CREATE DATABASE IF NOT EXISTS stock_db;
USE stock_db;

-- 创建外部表，结构与导入的 AAPL_stock_dividends.csv 严格对应
CREATE EXTERNAL TABLE IF NOT EXISTS stock_dividends_raw (
    event_date STRING,
    open_price DOUBLE,
    high_price DOUBLE,
    low_price DOUBLE,
    close_price DOUBLE,
    volume BIGINT,
    dividends DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',' -- CSV文件以逗号分隔
STORED AS TEXTFILE
TBLPROPERTIES ("skip.header.line.count"="1"); -- 跳过CSV文件的第一行表头

-- 将共享文件夹中的真实数据集加载到 Hive 表中
LOAD DATA LOCAL INPATH '/mnt/hgfs/bigdata_course_design/data/AAPL_stock_dividends.csv'
INTO TABLE stock_dividends_raw;