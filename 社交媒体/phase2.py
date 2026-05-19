import pandas as pd
import networkx as nx
import ast
import itertools


def main():
    # 加载第一阶段处理后的数据
    print("正在加载数据...")
    df = pd.read_csv('processed_chatgpt_data.csv')

    # 解析Hashtag列
    def parse_hashtags(tag_str):
        if pd.isna(tag_str) or tag_str == '[]':
            return []
        try:
            return ast.literal_eval(tag_str)
        except:
            return []

    print("提取话题关联...")
    df['hashtag_list'] = df['hashtag'].apply(parse_hashtags)

    # 构建网络
    G = nx.Graph()

    for tags in df['hashtag_list']:
        if len(tags) > 1:
            clean_tags = [t.replace('#', '').lower() for t in tags]
            # 对每条推文中的话题进行两两组合，建立连边
            edges = itertools.combinations(clean_tags, 2)
            for u, v in edges:
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    print(f"网络构建完成！节点数: {G.number_of_nodes()}, 连边数: {G.number_of_edges()}")

    # 网络特征计算
    degree_cent = nx.degree_centrality(G)
    top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n--- 核心话题排名 (度中心性) ---")
    for node, score in top_nodes:
        print(f"话题: {node}, 中心度: {score:.4f}")

    # 导出网络文件 (供Gephi使用)
    nx.write_gexf(G, "hashtag_network.gexf")
    print("\n网络文件已保存为: hashtag_network.gexf")


if __name__ == "__main__":
    main()
