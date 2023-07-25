from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from joblib import Parallel, delayed
import time

INDEX_NAME = "sample_index"


def create_mappings():
    """マッピング作成"""
    return {
        "mappings": {
            "properties": {
                "id": {"type": "integer"},
                "title": {
                    "type": "text",
                },
                "title_vector": {"type": "dense_vector", "dims": 128},
            }
        }
    }


def initialize_client(host="localhost", port=9200):
    """Elasticsearchの初期化"""
    uri = f"http://{host}:{port}"
    client = Elasticsearch(uri)
    # 既に存在していたら削除
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)

    mapping = create_mappings()
    client.indices.create(index=INDEX_NAME, body=mapping)
    return client


def finalize_client(client):
    """ElasticSearchの接続解除"""
    client.close()


def generate_sample_data(vector):
    return {
        "title": "test_title",
        "title_vector": vector.tolist(),
    }


def generate_sample_datas(vectors):
    for i, vector in enumerate(vectors):
        yield {
            "_index": INDEX_NAME,
            "title": f"title_{i}",
            "title_vector": vector.tolist(),
        }


def insert_index(client: Elasticsearch, index_name: str, body):
    client.index(index=index_name, body=body)


def search_index(client: Elasticsearch, index_name: str, vector, size=None):
    query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.title_vector, 'title_vector') + 1.0",
                "params": {"title_vector": vector.tolist()},
            },
        }
    }

    return client.search(
        index=index_name,
        body={
            "query": query,
            # "_source": {"includes": ["title", "title_vector"]},
        },
        size=size,
    )


if __name__ == "__main__":
    es = initialize_client()
    print(es.indices.get_mapping())

    indices = es.cat.indices(index="*", h="index").splitlines()
    for index in indices:
        print(index)

    # ここではベクトルはランダム値で代用する
    vector_num = 10000
    vector_size = 128
    vectors = np.random.uniform(low=0.0, high=1.0, size=(vector_num, vector_size))

    # 複数件まとめて登録する
    bulk(es, generate_sample_datas(vectors))
    # print(es.count(index=INDEX_NAME))

    # 1件登録する
    # vector = np.random.uniform(low=0.0, high=1.0, size=(vector_size,))
    # insert_index(es, INDEX_NAME, generate_sample_data(vector))

    # 検索用データの作成
    query_vector = np.random.uniform(low=0.0, high=1.0, size=(vector_size,))
    # 検索実行
    start_time = time.perf_counter_ns()
    res = search_index(es, INDEX_NAME, query_vector, size=3)
    end_time = time.perf_counter_ns()

    # print(res)
    top_object = res["hits"]["hits"][0]["_source"]
    top_df = pd.DataFrame(res["hits"]["hits"])
    print(top_df)
    print(top_object["title"])
    print("Search time by ElasticSearch: ", (end_time - start_time) / 1000000, "[ms]")

    # リスト内包表記で検索する場合の時間を計測
    start_time = time.perf_counter_ns()
    cos_sim = [1 - cosine(x, query_vector) for x in vectors]
    cos_sim[np.argmax(cos_sim)]
    end_time = time.perf_counter_ns()
    print("Search time by list: ", (end_time - start_time) / 1000000, "[ms]")

    # Paralellで検索する場合の時間を計測
    start_time = time.perf_counter_ns()
    _ = Parallel(n_jobs=-1)([delayed(cosine)(x, query_vector) for x in vectors])
    end_time = time.perf_counter_ns()
    print("Search time by Parallel: ", (end_time - start_time) / 1000000, "[ms]")

    finalize_client(es)
