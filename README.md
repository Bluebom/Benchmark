# 1. Extração (com cache)
AMOSTRA=100 python3 extracao.py

# 2. Embeddings (com cache)
AMOSTRA=100 BATCH_SIZE=2 python3 embedding.py

# 3. Indexação (quando tiver Elasticsearch)
python3 indexing.py
```

## 📁 Estrutura de Dados
```
data/
├── legislacao/     # PDFs originais
├── textos/         # JSONs extraídos (cache)
├── embeddings/     # .npy com vetores (cache)
└── resultados/     # Logs dos benchmarks