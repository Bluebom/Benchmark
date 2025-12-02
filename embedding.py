import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch

# Tentar carregar .env se existir
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# CONFIGURAÇÃO (via variáveis de ambiente ou valores padrão)
# =============================================================================
AMOSTRA = int(os.getenv("AMOSTRA", "0"))  # 0 = todos
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))  # H100 suporta mais que L40S
DIR_TEXTOS = Path(os.getenv("DIR_TEXTOS", "/workspace/tce-test/data/textos"))
DIR_EMBEDDINGS = Path(os.getenv("DIR_EMBEDDINGS", "/workspace/tce-test/data/embeddings"))
DIR_RESULTADOS = Path(os.getenv("DIR_RESULTADOS", "/workspace/tce-test/data/resultados"))
MODELO = os.getenv("MODELO", "Qwen/Qwen3-Embedding-8B")

# Constantes TCE-PB
TOTAL_LEGADO = 14_922_424
TOTAL_MENSAL = 181_412


def get_gpu_info() -> dict:
    """Retorna informações detalhadas da GPU."""
    if not torch.cuda.is_available():
        return {"disponivel": False, "nome": "CPU"}
    
    props = torch.cuda.get_device_properties(0)
    
    return {
        "disponivel": True,
        "nome": torch.cuda.get_device_name(0),
        "vram_total_gb": round(props.total_memory / (1024**3), 2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "compute_capability": f"{props.major}.{props.minor}"
    }


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
    
    print("=" * 70)
    print("🧠 GERAÇÃO DE EMBEDDINGS (QWEN3-8B) - H100")
    print("=" * 70)
    
    # GPU Info
    gpu_info = get_gpu_info()
    gpu_name = gpu_info.get("nome", "CPU")
    
    if gpu_info.get("disponivel"):
        vram = gpu_info.get("vram_total_gb", 0)
        print(f"\n🎮 GPU: {gpu_name}")
        print(f"   VRAM: {vram} GB")
        print(f"   CUDA: {gpu_info.get('cuda_version', 'N/A')}")
        print(f"   PyTorch: {gpu_info.get('pytorch_version', 'N/A')}")
        torch.cuda.empty_cache()
    else:
        print("\n⚠️  GPU não disponível, usando CPU")
    
    print(f"\n📂 DIR_TEXTOS: {DIR_TEXTOS}")
    print(f"📂 DIR_EMBEDDINGS: {DIR_EMBEDDINGS}")
    print(f"📂 DIR_RESULTADOS: {DIR_RESULTADOS}")
    
    # Carregar textos extraídos
    arquivos_json = list(DIR_TEXTOS.glob("*.json"))
    # Filtrar erros
    arquivos_json = [a for a in arquivos_json if "_ERRO" not in a.name]
    
    if AMOSTRA > 0:
        arquivos_json = arquivos_json[:AMOSTRA]
    
    print(f"\n📁 Textos disponíveis: {len(arquivos_json)}")
    
    if not arquivos_json:
        print("❌ Nenhum texto encontrado!")
        print(f"   Execute primeiro: python extracao.py")
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
        # Salvar resultado mesmo com cache
        resultado = {
            "data": datetime.now().isoformat(),
            "etapa": "embedding",
            "gpu": gpu_name,
            "modelo": MODELO,
            "batch_size": BATCH_SIZE,
            "docs_processados": 0,
            "cache": cache_count,
            "nota": "Todos do cache"
        }
        arq = DIR_RESULTADOS / f"embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(arq, "w") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"💾 Log salvo: {arq}")
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
    print(f"📊 Média tokens/doc: {total_tokens // len(docs):,}")
    
    # Carregar modelo
    print(f"\n⏳ Carregando {MODELO}...")
    t_load_start = time.perf_counter()
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODELO, model_kwargs={"torch_dtype": torch.float16})
    
    t_load = time.perf_counter() - t_load_start
    print(f"✅ Modelo carregado em {t_load:.1f}s")
    
    if torch.cuda.is_available():
        vram_modelo = torch.cuda.memory_allocated() / (1024**3)
        print(f"   VRAM após load: {vram_modelo:.2f} GB")
    
    print(f"\n📦 Batch size: {BATCH_SIZE}")
    
    torch.cuda.empty_cache()
    
    # Warm-up
    print("\n🔥 Warmup...")
    _ = model.encode(["teste de warmup"], batch_size=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Gerar embeddings
    print(f"\n⚡ Gerando embeddings...")
    t0 = time.perf_counter()
    
    textos = [d["texto"] for d in docs]
    embeddings = model.encode(
        textos,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_emb = time.perf_counter() - t0
    
    # VRAM pico
    vram_pico = 0
    if torch.cuda.is_available():
        vram_pico = torch.cuda.max_memory_allocated() / (1024**3)
    
    # Salvar embeddings individuais
    print("\n💾 Salvando embeddings...")
    for doc, emb in tqdm(zip(docs, embeddings), total=len(docs), desc="Salvando", unit="emb"):
        salvar_embedding(doc["id"], emb)
    
    # Métricas
    docs_hora = len(docs) / t_emb * 3600 if t_emb > 0 else 0
    tok_s = total_tokens / t_emb if t_emb > 0 else 0
    ms_por_doc = (t_emb / len(docs)) * 1000 if len(docs) > 0 else 0
    
    # Projeções
    h_leg = TOTAL_LEGADO / docs_hora if docs_hora > 0 else float('inf')
    h_men = TOTAL_MENSAL / docs_hora if docs_hora > 0 else float('inf')
    
    # Resumo
    print(f"""
{'=' * 70}
📊 RESUMO DOS EMBEDDINGS
{'=' * 70}

┌─────────────────────────────────────────────────────────────────────┐
│  CONFIGURAÇÃO                                                        │
├─────────────────────────────────────────────────────────────────────┤
│  🎮 GPU:                  {gpu_name:<40} │
│  🧠 Modelo:               {MODELO:<40} │
│  📦 Batch size:           {BATCH_SIZE:<40} │
│  📐 Dimensões:            {embeddings.shape[1]:<40} │
├─────────────────────────────────────────────────────────────────────┤
│  PROCESSAMENTO                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  📝 Documentos:           {len(docs):>10,}                              │
│  📦 Cache (já existiam):  {cache_count:>10,}                              │
│  🔢 Tokens processados:   {total_tokens:>10,}                              │
├─────────────────────────────────────────────────────────────────────┤
│  MÉTRICAS                                                            │
├─────────────────────────────────────────────────────────────────────┤
│  ⏱️  Tempo total:          {t_emb:>10.2f}s                             │
│  ⏱️  Tempo médio/doc:      {ms_por_doc:>10.2f}ms                            │
│  📈 Throughput docs:      {docs_hora:>10,.0f} docs/hora                   │
│  🔢 Throughput tokens:    {tok_s:>10,.0f} tokens/segundo              │
│  💾 VRAM pico:            {vram_pico:>10.2f} GB                            │
├─────────────────────────────────────────────────────────────────────┤
│  🔮 PROJEÇÕES (só embedding)                                         │
├─────────────────────────────────────────────────────────────────────┤
│  📚 Legado (14.9M docs):  {h_leg:>10,.0f}h ({h_leg/24:,.0f} dias)              │
│  📅 Mensal (181K docs):   {h_men:>10.1f}h                              │
└─────────────────────────────────────────────────────────────────────┘

📂 Embeddings salvos em: {DIR_EMBEDDINGS}
""")
    
    # Salvar resultado detalhado
    resultado = {
        "data": datetime.now().isoformat(),
        "etapa": "embedding",
        "gpu": gpu_name,
        "gpu_info": gpu_info,
        "modelo": MODELO,
        "config": {
            "batch_size": BATCH_SIZE,
            "amostra": AMOSTRA,
            "dimensoes": int(embeddings.shape[1])
        },
        "processamento": {
            "docs_processados": len(docs),
            "cache": cache_count,
            "tokens_processados": total_tokens,
            "media_tokens_por_doc": total_tokens // len(docs) if len(docs) > 0 else 0
        },
        "metricas": {
            "tempo_total_s": round(t_emb, 2),
            "tempo_carga_modelo_s": round(t_load, 2),
            "tempo_medio_por_doc_ms": round(ms_por_doc, 2),
            "docs_hora": round(docs_hora, 0),
            "tokens_segundo": round(tok_s, 0),
            "vram_pico_gb": round(vram_pico, 2)
        },
        "projecoes": {
            "legado_docs": TOTAL_LEGADO,
            "legado_horas": round(h_leg, 0),
            "legado_dias": round(h_leg / 24, 0),
            "mensal_docs": TOTAL_MENSAL,
            "mensal_horas": round(h_men, 1)
        },
        "diretorio_saida": str(DIR_EMBEDDINGS)
    }
    
    arq = DIR_RESULTADOS / f"embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(arq, "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Métricas salvas em: {arq}")


if __name__ == "__main__":
    main()