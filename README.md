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
