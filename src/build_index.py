from gensim.models import KeyedVectors
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from swem import MeCabTokenizer
from swem import SWEM
import json
import gzip
from joblib import Parallel, delayed
from pathos.multiprocessing import ProcessPool

W2V_PATH = "./src/wiki/jawiki.word_vectors.200d.txt"
INDEX_NAME = "wikipedia_index"
BATCH_SIZE = 1000


def initialize_client(host="localhost", port=9200):
    """Elasticsearchの初期化"""
    uri = f"http://{host}:{port}"
    client = Elasticsearch(uri)

    return client


def index_batch(client, docs):
    pool = ProcessPool(nodes=6)
    requests = pool.map(get_request, docs)
    # requests = Parallel(n_jobs=-1)([delayed(get_request)(doc) for doc in docs])
    bulk(client, requests)


def get_request(doc):
    return {
        "_op_type": "index",
        "_index": INDEX_NAME,
        "text": doc["text"],
        "title": doc["title"],
        "text_vector": swem.average_pooling(doc["text"]).tolist(),
    }


if __name__ == "__main__":
    # エンベディングの読み込み
    w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)
    tokenizer = MeCabTokenizer("")
    swem = SWEM(w2v, tokenizer)

    # ElastigSearchの初期化
    es = initialize_client()

    # インデックスの作成
    # 既に存在していたら削除
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    config = json.load(open("./src/index.json"))
    es.indices.create(
        index=INDEX_NAME, settings=config["settings"], mappings=config["mappings"]
    )
    print(es.indices.get_mapping(index=INDEX_NAME))

    docs = []
    count = 0
    with gzip.open("./src/wiki/jawiki-20230522-cirrussearch-content.json.gz") as f:
        for line in f:
            json_line = json.loads(line)
            if "index" not in json_line:
                doc = json_line
                docs.append(doc)
                count += 1

                if count % BATCH_SIZE == 0:
                    index_batch(es, docs)
                    docs = []
                    print(f"Indexed {count} documents. {100.0*count/1165654}%")
        if docs:
            index_batch(es, docs)
            print(f"Indexed {count} documents.")
