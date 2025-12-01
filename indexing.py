import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Configuração
AMOSTRA = int(os.getenv("AMOSTRA", "0"))  # 0 = todos
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
NOME_INDICE = os.getenv("NOME_INDICE", "tce_legislacao")
DIR_TEXTOS = Path(os.getenv("DIR_TEXTOS", "/workspace/tce-test/data/textos"))
DIR_EMBEDDINGS = Path(os.getenv("DIR_EMBEDDINGS", "/workspace/tce-test/data/embeddings"))
DIR_RESULTADOS = Path(os.getenv("DIR_RESULTADOS", "/workspace/tce-test/data/resultados"))


def main():
    print("="*60)
    print("INDEXAÇÃO NO ELASTICSEARCH")
    print("="*60)
    
    # Verificar arquivos disponíveis
    textos_json = list(DIR_TEXTOS.glob("*.json"))
    textos_json = [t for t in textos_json if "_ERRO" not in t.name]
    embeddings_npy = list(DIR_EMBEDDINGS.glob("*.npy"))
    
    print(f"\n📁 Textos disponíveis: {len(textos_json)}")
    print(f"📁 Embeddings disponíveis: {len(embeddings_npy)}")
    
    # Encontrar documentos com texto E embedding
    textos_ids = {t.stem for t in textos_json}
    embeddings_ids = {e.stem for e in embeddings_npy}
    prontos = textos_ids & embeddings_ids
    
    print(f"✅ Prontos para indexar: {len(prontos)}")
    
    if not prontos:
        print("❌ Nenhum documento pronto! Execute extracao.py e embedding.py primeiro.")
        return
    
    # Tentar conectar ao Elasticsearch
    try:
        from elasticsearch import Elasticsearch, helpers
        
        es = Elasticsearch(ES_HOST)
        if not es.ping():
            raise Exception("Elasticsearch não respondeu")
        
        print(f"\n✅ Conectado ao Elasticsearch: {ES_HOST}")
        
    except Exception as e:
        print(f"""
⚠️  Elasticsearch não disponível: {e}

Para usar indexação, inicie o Elasticsearch:
  docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.15.0

Ou configure ES_HOST com o endereço correto.
""")
        
        # Mostrar preview dos dados
        print("\n📋 Preview dos dados prontos para indexar:")
        for doc_id in list(prontos)[:3]:
            texto_file = DIR_TEXTOS / f"{doc_id}.json"
            emb_file = DIR_EMBEDDINGS / f"{doc_id}.npy"
            
            with open(texto_file, "r") as f:
                texto_data = json.load(f)
            emb = np.load(emb_file)
            
            print(f"\n  📄 {doc_id}")
            print(f"     Texto: {len(texto_data['texto'])} chars")
            print(f"     Embedding: {emb.shape}")
        
        return
    
    # Criar índice se não existir
    if not es.indices.exists(index=NOME_INDICE):
        print(f"\n📝 Criando índice '{NOME_INDICE}'...")
        
        mapeamento = {
            "settings": {
                "number_of_shards": 3,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "id_documento": {"type": "keyword"},
                    "nome_arquivo": {"type": "keyword"},
                    "caminho_original": {"type": "keyword"},
                    "texto": {"type": "text", "analyzer": "portuguese"},
                    "texto_resumo": {"type": "text", "analyzer": "portuguese"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 4096,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "tamanho_bytes": {"type": "long"},
                    "tokens_estimados": {"type": "integer"},
                    "data_extracao": {"type": "date"},
                    "data_indexacao": {"type": "date"}
                }
            }
        }
        
        es.indices.create(index=NOME_INDICE, body=mapeamento)
        print(f"✅ Índice criado!")
    
    # Verificar já indexados
    if AMOSTRA > 0:
        prontos = list(prontos)[:AMOSTRA]
    else:
        prontos = list(prontos)
    
    # Preparar documentos para bulk
    print(f"\n⏳ Preparando {len(prontos)} documentos...")
    
    acoes = []
    for doc_id in tqdm(prontos, desc="Preparando"):
        texto_file = DIR_TEXTOS / f"{doc_id}.json"
        emb_file = DIR_EMBEDDINGS / f"{doc_id}.npy"
        
        with open(texto_file, "r") as f:
            texto_data = json.load(f)
        emb = np.load(emb_file)
        
        acao = {
            "_index": NOME_INDICE,
            "_id": doc_id,
            "_source": {
                "id_documento": doc_id,
                "nome_arquivo": texto_data["nome_arquivo"],
                "caminho_original": texto_data["caminho_original"],
                "texto": texto_data["texto"],
                "texto_resumo": texto_data["texto"][:500],
                "embedding": emb.tolist(),
                "tamanho_bytes": texto_data["tamanho_bytes"],
                "tokens_estimados": texto_data["tokens_estimados"],
                "data_extracao": texto_data["data_extracao"],
                "data_indexacao": datetime.now().isoformat()
            }
        }
        acoes.append(acao)
    
    # Bulk insert
    print(f"\n⏳ Indexando {len(acoes)} documentos...")
    
    t0 = time.time()
    sucesso, falhas = helpers.bulk(es, acoes, raise_on_error=False)
    es.indices.refresh(index=NOME_INDICE)
    t_idx = time.time() - t0
    
    docs_hora = sucesso / t_idx * 3600 if t_idx > 0 else 0
    erros = len(falhas) if isinstance(falhas, list) else 0
    
    print(f"""
{'='*60}
RESUMO DA INDEXAÇÃO
{'='*60}
┌─────────────────────────────────────────────────────────┐
│  Elasticsearch: {ES_HOST}
│  Índice: {NOME_INDICE}
├─────────────────────────────────────────────────────────┤
│  ✅ Indexados: {sucesso}
│  ❌ Erros: {erros}
├─────────────────────────────────────────────────────────┤
│  ⏱️  Tempo: {t_idx:.1f}s
│  📈 Throughput: {docs_hora:,.0f} docs/hora
└─────────────────────────────────────────────────────────┘
""")
    
    # Salvar resultado
    resultado = {
        "data": datetime.now().isoformat(),
        "etapa": "indexacao",
        "es_host": ES_HOST,
        "indice": NOME_INDICE,
        "docs_indexados": sucesso,
        "erros": erros,
        "tempo_s": round(t_idx, 2),
        "docs_hora": round(docs_hora, 0)
    }
    
    arq = DIR_RESULTADOS / f"indexacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(arq, "w") as f:
        json.dump(resultado, f, indent=2)
    
    print(f"💾 Log salvo: {arq}")


if __name__ == "__main__":
    main()