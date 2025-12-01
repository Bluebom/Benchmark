import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
from dotenv import load_dotenv
load_dotenv()

# Configuração
AMOSTRA = int(os.getenv("AMOSTRA", "0"))  # 0 = todos
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
DIR_TEXTOS = Path(os.getenv("DIR_TEXTOS", "/workspace/tce-test/data/textos"))
DIR_EMBEDDINGS = Path(os.getenv("DIR_EMBEDDINGS", "/workspace/tce-test/data/embeddings"))
DIR_RESULTADOS = Path(os.getenv("DIR_RESULTADOS", "/workspace/tce-test/data/resultados"))
MODELO = os.getenv("MODELO", "Qwen/Qwen3-Embedding-8B")

TOTAL_LEGADO = 14_922_424
TOTAL_MENSAL = 181_412


def carregar_embedding_existente(doc_id: str) -> bool:
    """Verifica se embedding já existe"""
    arquivo_npy = DIR_EMBEDDINGS / f"{doc_id}.npy"
    return arquivo_npy.exists()


def salvar_embedding(doc_id: str, embedding: np.ndarray):
    """Salva embedding em arquivo .npy"""
    arquivo_npy = DIR_EMBEDDINGS / f"{doc_id}.npy"
    np.save(arquivo_npy, embedding)


def main():
    DIR_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GERAÇÃO DE EMBEDDINGS (QWEN3-8B)")
    print("="*60)
    
    # GPU Info
    gpu_name = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🖥️  GPU: {gpu_name} ({vram:.0f}GB)")
        torch.cuda.empty_cache()
    
    # Carregar textos extraídos
    arquivos_json = list(DIR_TEXTOS.glob("*.json"))
    # Filtrar erros
    arquivos_json = [a for a in arquivos_json if "_ERRO" not in a.name]
    
    if AMOSTRA > 0:
        arquivos_json = arquivos_json[:AMOSTRA]
    
    print(f"\n📁 Textos disponíveis: {len(arquivos_json)}")
    
    if not arquivos_json:
        print("❌ Nenhum texto encontrado! Execute extracao.py primeiro.")
        return
    
    # Verificar cache de embeddings
    novos = []
    cache_count = 0
    
    for arq in arquivos_json:
        doc_id = arq.stem
        if carregar_embedding_existente(doc_id):
            cache_count += 1
        else:
            novos.append(arq)
    
    print(f"✅ Já processados (cache): {cache_count}")
    print(f"🆕 A processar: {len(novos)}")
    
    if not novos:
        print("\n✅ Todos os embeddings já foram gerados!")
        return
    
    # Carregar documentos
    docs = []
    total_tokens = 0
    
    for arq in novos:
        with open(arq, "r", encoding="utf-8") as f:
            data = json.load(f)
            docs.append({
                "id": data["id"],
                "texto": data["texto"],
                "tokens": data.get("tokens_estimados", len(data["texto"]) // 4)
            })
            total_tokens += docs[-1]["tokens"]
    
    print(f"\n📝 Documentos a processar: {len(docs)}")
    print(f"🔢 Tokens total: {total_tokens:,}")
    
    # Carregar modelo
    print(f"\n⏳ Carregando {MODELO}...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(MODELO, model_kwargs={"torch_dtype": torch.float16})
    print("✅ Modelo carregado!")
    
    print(f"📦 Batch size: {BATCH_SIZE}")
    
    torch.cuda.empty_cache()
    
    # Warm-up
    _ = model.encode(["teste"], batch_size=1)
    torch.cuda.synchronize()
    
    # Gerar embeddings
    t0 = time.time()
    
    textos = [d["texto"] for d in docs]
    embeddings = model.encode(
        textos,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    torch.cuda.synchronize()
    t_emb = time.time() - t0
    
    # Salvar embeddings individuais
    print("\n💾 Salvando embeddings...")
    for doc, emb in zip(docs, embeddings):
        salvar_embedding(doc["id"], emb)
    
    # Métricas
    docs_hora = len(docs) / t_emb * 3600 if t_emb > 0 else 0
    tok_s = total_tokens / t_emb if t_emb > 0 else 0
    
    # Projeções
    h_leg = TOTAL_LEGADO / docs_hora if docs_hora > 0 else 0
    h_men = TOTAL_MENSAL / docs_hora if docs_hora > 0 else 0
    
    print(f"""
{'='*60}
RESUMO DOS EMBEDDINGS
{'='*60}
┌─────────────────────────────────────────────────────────┐
│  GPU: {gpu_name}
│  Modelo: {MODELO}
│  Batch size: {BATCH_SIZE}
├─────────────────────────────────────────────────────────┤
│  📝 Documentos processados: {len(docs)}
│  📦 Cache (já existiam): {cache_count}
│  🔢 Tokens: {total_tokens:,}
│  📐 Dimensões: {embeddings.shape[1]}
├─────────────────────────────────────────────────────────┤
│  ⏱️  Tempo: {t_emb:.1f}s
│  📈 Throughput: {docs_hora:,.0f} docs/hora
│  🔢 Tokens/seg: {tok_s:,.0f}
├─────────────────────────────────────────────────────────┤
│  PROJEÇÕES (só embedding)
│    Legado (14.9M): {h_leg:,.0f}h ({h_leg/24:.0f} dias)
│    Mensal (181K): {h_men:.1f}h
├─────────────────────────────────────────────────────────┤
│  📂 Embeddings salvos em: {DIR_EMBEDDINGS}
└─────────────────────────────────────────────────────────┘
""")
    
    # Salvar resultado
    resultado = {
        "data": datetime.now().isoformat(),
        "etapa": "embedding",
        "gpu": gpu_name,
        "modelo": MODELO,
        "batch_size": BATCH_SIZE,
        "docs_processados": len(docs),
        "cache": cache_count,
        "tokens": total_tokens,
        "dimensoes": int(embeddings.shape[1]),
        "tempo_s": round(t_emb, 2),
        "docs_hora": round(docs_hora, 0),
        "tokens_segundo": round(tok_s, 0),
        "projecoes": {
            "legado_horas": round(h_leg, 0),
            "legado_dias": round(h_leg/24, 0),
            "mensal_horas": round(h_men, 1)
        },
        "diretorio_saida": str(DIR_EMBEDDINGS)
    }
    
    arq = DIR_RESULTADOS / f"embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(arq, "w") as f:
        json.dump(resultado, f, indent=2)
    
    print(f"💾 Log salvo: {arq}")


if __name__ == "__main__":
    main()