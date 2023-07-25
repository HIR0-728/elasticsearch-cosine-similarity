# Elastic Search で色々検証してみるようリポジトリ

## Requirements

- docker
- docker-compose

## Setup

```bash
docker-compose build
docker-compose up -d

# ElasticSearchへの接続確認
curl -X GET "localhost:9200/_cat/health?v&pretty"

# Python実行環境の構築
poetry install
```

## Start

```bash
# コサイン類似度検索のテストコード
python src/main.py

# Wikipediaデータ取得
cd src/wiki
wget https://github.com/singletongue/WikiEntVec/releases/download/20190520/jawiki.word_vectors.200d.txt.bz2
wget https://dumps.wikimedia.org/other/cirrussearch/20230522/jawiki-20230522-cirrussearch-content.json.gz

# Wikipediaデータのインデックス作成
cd ../../
python src/build_index.py

# Wikipediaデータを使った類似度検索の実行
python src/search_wiki.py


```
