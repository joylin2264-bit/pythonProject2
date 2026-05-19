import pandas as pd
import re
import demoji
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

def clean_tweet_advanced(text):
    text = str(text)
    # 将 Emoji 转换为文字描述
    text = demoji.replace_with_desc(text, sep=" ")
    # 清洗链接
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 清洗@用户和#号
    text = re.sub(r'\@\w+|\#', '', text)
    # 去除多余空格和换行
    text = text.replace('\n', ' ').strip()
    return text


def get_sentiment(text):
    # 计算情感得分：针对英文。非英文返回 0.0
    return TextBlob(text).sentiment.polarity


def main():
    # 加载数据
    file_path = 'chatgpt1.csv'
    print("正在加载数据...")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='latin1')

    # 转换时间列
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime'])  # 剔除时间损坏的行

    # 预处理
    print(f"执行数据预处理（共 {len(df)} 条）...")
    df['Clean_Text'] = df['Text'].apply(clean_tweet_advanced)

    # 情感分析
    print("正在分析情感倾向...")
    df['Sentiment_Score'] = df['Clean_Text'].apply(get_sentiment)
    df['Sentiment_Type'] = df['Sentiment_Score'].apply(
        lambda s: 'Positive' if s > 0 else ('Negative' if s < 0 else 'Neutral')
    )

    # 可视化
    print("正在生成词云图...")
    all_words = ' '.join([text for text in df['Clean_Text']])
    my_stopwords = set(STOPWORDS)
    # 过滤干扰词
    my_stopwords.update(['chatgpt', 'ai', 'bot', 'use', 'will', 'new', 'now'])

    wordcloud = WordCloud(
        width=1000, height=600,
        background_color='white',
        stopwords=my_stopwords,
        max_words=100,
        colormap='tab10'
    ).generate(all_words)

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('ChatGPT Social Media Discussion - Key Topics', fontsize=20)
    plt.show()

    print("\n--- 实验阶段 1 统计结果 ---")
    print(df['Sentiment_Type'].value_counts())

    # 保存结果
    output_file = 'processed_chatgpt_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n处理后的数据已保存至: {output_file}")


if __name__ == "__main__":
    main()
