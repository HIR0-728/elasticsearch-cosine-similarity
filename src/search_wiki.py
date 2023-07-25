from gensim.models import KeyedVectors
from elasticsearch import Elasticsearch
import time
from swem import MeCabTokenizer
from swem import SWEM


W2V_PATH = "./src/wiki/jawiki.word_vectors.200d.txt"
INDEX_NAME = "wikipedia_index"
BATCH_SIZE = 1000
SEARCH_SIZE = 10


def initialize_client(host="localhost", port=9200):
    """Elasticsearchの初期化"""
    uri = f"http://{host}:{port}"
    client = Elasticsearch(uri)

    return client


def handle_query(client, swem):
    query = input("Enter query text: ")
    embedding_start_time = time.time()
    query_vector = swem.average_pooling(query).tolist()
    embedding_end_time = time.time() - embedding_start_time

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                "params": {"query_vector": query_vector},
            },
        }
    }

    search_start_time = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "text"]},
        },
    )
    search_end_time = time.time() - search_start_time

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("encoding time: {:.2f} ms".format(embedding_end_time * 1000))
    print("search time: {:.2f} ms".format(search_end_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"]["title"])
        print(hit["_source"]["text"][:200])
        print()


if __name__ == "__main__":
    # エンベディングの読み込み
    w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)
    tokenizer = MeCabTokenizer("")
    swem = SWEM(w2v, tokenizer)

    # ElastigSearchの初期化
    es = initialize_client()

    # 検索ループ
    while True:
        try:
            handle_query(es, swem)
        except KeyboardInterrupt:
            break
