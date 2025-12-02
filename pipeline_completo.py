import os
import time
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
from dotenv import load_dotenv

# Tentar carregar .env se existir
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

AMOSTRA = int(os.getenv("AMOSTRA", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "16000"))
DIR_PDFS = Path(os.getenv("DIR_PDFS", "/workspace/tce-test/data/legislacao"))
DIR_RESULTADOS = Path(os.getenv("DIR_RESULTADOS", "/workspace/tce-test/data/resultados"))
MODELO = os.getenv("MODELO", "Qwen/Qwen3-Embedding-8B")

TOTAL_LEGADO = 14_922_424
TOTAL_MENSAL = 181_412


def extrair_pymupdf(pdf_path, max_chars=16000):
    """Extrai texto com PyMuPDF (muito mais rápido)"""
    import fitz
    try:
        doc = fitz.open(str(pdf_path))
        texto = ""
        for page in doc:
            texto += page.get_text()
            if len(texto) >= max_chars:
                break
        doc.close()
        return texto[:max_chars]
    except:
        return ""


def main():
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("BENCHMARK L40S - PyMuPDF")
    print("="*60)
    
    gpu_name = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🖥️  GPU: {gpu_name} ({vram:.0f}GB)")
        torch.cuda.empty_cache()
    
    pdfs = list(DIR_PDFS.rglob("*.pdf"))[:AMOSTRA]
    print(f"📁 PDFs: {len(pdfs)}")
    
    # === EXTRAÇÃO PyMuPDF ===
    print("\n" + "="*60)
    print("ETAPA 1: EXTRAÇÃO (PyMuPDF)")
    print("="*60)
    
    docs = []
    total_bytes = 0
    total_tokens = 0
    erros = 0
    
    t0 = time.time()
    
    for pdf in tqdm(pdfs, desc="Extraindo"):
        texto = extrair_pymupdf(pdf, MAX_CHARS)
        if texto:
            tokens = len(texto) // 4
            docs.append({"texto": texto, "tokens": tokens})
            total_bytes += pdf.stat().st_size
            total_tokens += tokens
        else:
            erros += 1
    
    t_ext = time.time() - t0
    docs_h_ext = len(docs) / t_ext * 3600 if t_ext > 0 else 0
    
    print(f"\n✅ {len(docs)} docs | ❌ {erros} erros")
    print(f"⏱️  {t_ext:.1f}s | 📈 {docs_h_ext:,.0f} docs/h")
    
    if len(docs) == 0:
        print("❌ Nenhum documento extraído!")
        return
    
    # === EMBEDDING ===
    print("\n" + "="*60)
    print("ETAPA 2: EMBEDDING (QWEN3-8B)")
    print("="*60)
    
    from sentence_transformers import SentenceTransformer
    
    print("Carregando modelo...")
    model = SentenceTransformer(MODELO, model_kwargs={"torch_dtype": torch.float16})
    print("✅ Modelo OK")
    
    print(f"\n📝 {len(docs)} textos | 🔢 {total_tokens:,} tokens | 📦 batch={BATCH_SIZE}")
    
    torch.cuda.empty_cache()
    _ = model.encode(["teste"], batch_size=1)
    torch.cuda.synchronize()
    
    t0 = time.time()
    
    embeddings = model.encode(
        [d["texto"] for d in docs],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    torch.cuda.synchronize()
    t_emb = time.time() - t0
    
    docs_h_emb = len(docs) / t_emb * 3600 if t_emb > 0 else 0
    tok_s = total_tokens / t_emb if t_emb > 0 else 0
    
    print(f"\n✅ {len(embeddings)} embeddings ({embeddings.shape[1]} dims)")
    print(f"⏱️  {t_emb:.1f}s | 📈 {docs_h_emb:,.0f} docs/h | 🔢 {tok_s:,.0f} tok/s")
    
    # === RESUMO ===
    gargalo_h = min(docs_h_ext, docs_h_emb)
    gargalo = "Extração" if gargalo_h == docs_h_ext else "Embedding"
    
    h_leg = TOTAL_LEGADO / gargalo_h if gargalo_h > 0 else 0
    h_men = TOTAL_MENSAL / gargalo_h if gargalo_h > 0 else 0
    
    speedup_vs_docling = docs_h_ext / 1831
    
    print(f"""
{'='*60}
RESUMO
{'='*60}
┌─────────────────────────────────────────────────────────┐
│  GPU: {gpu_name}
│  Extrator: PyMuPDF
│  Amostra: {len(docs)} docs ({total_bytes/(1024*1024):.2f} MB)
│  Tokens: {total_tokens:,}
├─────────────────────────────────────────────────────────┤
│  EXTRAÇÃO:  {t_ext:.1f}s → {docs_h_ext:,.0f} docs/hora
│             Speedup vs Docling: {speedup_vs_docling:.1f}x
│  EMBEDDING: {t_emb:.1f}s → {docs_h_emb:,.0f} docs/hora
│             {tok_s:,.0f} tokens/segundo
├─────────────────────────────────────────────────────────┤
│  GARGALO: {gargalo} ({gargalo_h:,.0f} docs/hora)
├─────────────────────────────────────────────────────────┤
│  PROJEÇÕES:
│    Legado (14.9M): {h_leg:,.0f}h ({h_leg/24:.0f} dias)
│    Mensal (181K):  {h_men:.1f}h
└─────────────────────────────────────────────────────────┘
""")
    
    # Salvar
    resultado = {
        "data": datetime.now().isoformat(),
        "gpu": gpu_name,
        "extrator": "PyMuPDF",
        "modelo": MODELO,
        "batch_size": BATCH_SIZE,
        "amostra": {"docs": len(docs), "tamanho_mb": round(total_bytes/(1024*1024), 2), "tokens": total_tokens},
        "extracao": {"tempo_s": round(t_ext, 2), "docs_hora": round(docs_h_ext, 0), "erros": erros, "speedup_vs_docling": round(speedup_vs_docling, 1)},
        "embedding": {"tempo_s": round(t_emb, 2), "docs_hora": round(docs_h_emb, 0), "tokens_segundo": round(tok_s, 0)},
        "gargalo": gargalo,
        "projecoes": {"legado_horas": round(h_leg, 0), "legado_dias": round(h_leg/24, 0), "mensal_horas": round(h_men, 1)}
    }
    
    arq = DIR_RESULTADOS / f"benchmark_pymupdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(arq, "w") as f:
        json.dump(resultado, f, indent=2)
    
    print(f"💾 Salvo: {arq}")

if __name__ == "__main__":
    main()