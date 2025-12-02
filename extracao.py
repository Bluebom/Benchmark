import os
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

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
MAX_CHARS = int(os.getenv("MAX_CHARS", "16000"))
DIR_PDFS = Path(os.getenv("DIR_PDFS", "/workspace/tce-test/data/legislacao"))
DIR_TEXTOS = Path(os.getenv("DIR_TEXTOS", "/workspace/tce-test/data/textos"))
DIR_RESULTADOS = Path(os.getenv("DIR_RESULTADOS", "/workspace/tce-test/data/resultados"))

# Constantes TCE-PB
TOTAL_LEGADO = 14_922_424
TOTAL_MENSAL = 181_412


def carregar_existente(pdf_path: Path) -> dict | None:
    """Carrega texto já extraído se existir"""
    arquivo_json = DIR_TEXTOS / f"{pdf_path.stem}.json"
    if arquivo_json.exists():
        try:
            with open(arquivo_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None


def salvar_extracao(pdf_path: Path, texto: str, texto_completo_len: int) -> dict:
    """Salva extração em JSON"""
    doc_data = {
        "id": pdf_path.stem,
        "nome_arquivo": pdf_path.name,
        "caminho_original": str(pdf_path),
        "texto": texto[:MAX_CHARS],
        "texto_completo_chars": texto_completo_len,
        "texto_truncado_chars": len(texto[:MAX_CHARS]),
        "tokens_estimados": len(texto[:MAX_CHARS]) // 4,
        "tamanho_bytes": pdf_path.stat().st_size,
        "data_extracao": datetime.now().isoformat()
    }
    
    arquivo_json = DIR_TEXTOS / f"{pdf_path.stem}.json"
    with open(arquivo_json, "w", encoding="utf-8") as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2)
    
    return doc_data


def salvar_erro(pdf_path: Path, erro: str):
    """Salva erro de extração"""
    erro_data = {
        "id": pdf_path.stem,
        "nome_arquivo": pdf_path.name,
        "caminho_original": str(pdf_path),
        "erro": erro,
        "data_extracao": datetime.now().isoformat()
    }
    arquivo_erro = DIR_TEXTOS / f"{pdf_path.stem}_ERRO.json"
    with open(arquivo_erro, "w", encoding="utf-8") as f:
        json.dump(erro_data, f, ensure_ascii=False, indent=2)


def main():
    DIR_TEXTOS.mkdir(parents=True, exist_ok=True)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📄 EXTRAÇÃO DE TEXTO (PyMuPDF)")
    print("=" * 70)
    print(f"   DIR_PDFS: {DIR_PDFS}")
    print(f"   DIR_TEXTOS: {DIR_TEXTOS}")
    print(f"   DIR_RESULTADOS: {DIR_RESULTADOS}")
    print(f"   MAX_CHARS: {MAX_CHARS:,}")
    print(f"   AMOSTRA: {AMOSTRA if AMOSTRA > 0 else 'TODOS'}")
    
    # Listar PDFs
    if not DIR_PDFS.exists():
        print(f"\n❌ Diretório não encontrado: {DIR_PDFS}")
        return
    
    pdfs = list(DIR_PDFS.rglob("*.pdf"))
    if AMOSTRA > 0:
        pdfs = pdfs[:AMOSTRA]
    
    print(f"\n📁 PDFs encontrados: {len(pdfs)}")
    
    if not pdfs:
        print("❌ Nenhum PDF encontrado!")
        return
    
    # Verificar cache
    novos = []
    cache = []
    
    for pdf in pdfs:
        existente = carregar_existente(pdf)
        if existente:
            cache.append(existente)
        else:
            novos.append(pdf)
    
    print(f"✅ Já extraídos (cache): {len(cache)}")
    print(f"🆕 A extrair: {len(novos)}")
    
    if not novos:
        print("\n✅ Todos os PDFs já foram extraídos!")
        # Salvar resultado mesmo com cache
        resultado = {
            "data": datetime.now().isoformat(),
            "etapa": "extracao",
            "total_pdfs": len(pdfs),
            "cache": len(cache),
            "novos_processados": 0,
            "extraidos": 0,
            "erros": 0,
            "tempo_s": 0,
            "docs_hora": 0,
            "nota": "Todos do cache"
        }
        arq = DIR_RESULTADOS / f"extracao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(arq, "w") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"💾 Log salvo: {arq}")
        return
    
    # Carregar PyMuPDF
    print("\n⏳ Carregando PyMuPDF...")
    t_load_start = time.perf_counter()

    import fitz  # PyMuPDF

    t_load = time.perf_counter() - t_load_start
    print(f"✅ PyMuPDF carregado em {t_load:.1f}s")
    
    # Extrair novos
    extraidos = 0
    erros = 0
    total_bytes = 0
    total_tokens = 0
    total_chars = 0
    
    t0 = time.perf_counter()
    
    for pdf in tqdm(novos, desc="Extraindo", unit="doc"):
        try:
            # Abrir PDF com PyMuPDF
            doc = fitz.open(str(pdf))
            texto_completo = ""

            # Extrair texto de todas as páginas
            for page in doc:
                texto_completo += page.get_text()

            doc.close()

            doc_data = salvar_extracao(pdf, texto_completo, len(texto_completo))

            extraidos += 1
            total_bytes += doc_data["tamanho_bytes"]
            total_tokens += doc_data["tokens_estimados"]
            total_chars += doc_data["texto_truncado_chars"]

        except Exception as e:
            salvar_erro(pdf, str(e))
            erros += 1
    
    t_total = time.perf_counter() - t0
    
    # Métricas
    docs_hora = extraidos / t_total * 3600 if t_total > 0 else 0
    segundos_por_doc = t_total / extraidos if extraidos > 0 else 0
    
    # Projeções
    h_legado = TOTAL_LEGADO / docs_hora if docs_hora > 0 else float('inf')
    h_mensal = TOTAL_MENSAL / docs_hora if docs_hora > 0 else float('inf')
    
    # Resumo
    print(f"""
{'=' * 70}
📊 RESUMO DA EXTRAÇÃO
{'=' * 70}

┌─────────────────────────────────────────────────────────────────────┐
│  PROCESSAMENTO                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  📁 PDFs processados:     {len(novos):>10,}                              │
│  ✅ Extraídos:            {extraidos:>10,}                              │
│  ❌ Erros:                {erros:>10,}                              │
│  📦 Cache (já existiam):  {len(cache):>10,}                              │
├─────────────────────────────────────────────────────────────────────┤
│  MÉTRICAS                                                            │
├─────────────────────────────────────────────────────────────────────┤
│  ⏱️  Tempo total:          {t_total:>10.1f}s                             │
│  ⏱️  Tempo médio/doc:      {segundos_por_doc:>10.2f}s                             │
│  📈 Throughput:           {docs_hora:>10,.0f} docs/hora                   │
├─────────────────────────────────────────────────────────────────────┤
│  DADOS                                                               │
├─────────────────────────────────────────────────────────────────────┤
│  💾 Tamanho PDFs:         {total_bytes/(1024*1024):>10.2f} MB                            │
│  📝 Caracteres:           {total_chars:>10,}                              │
│  🔢 Tokens (est.):        {total_tokens:>10,}                              │
├─────────────────────────────────────────────────────────────────────┤
│  🔮 PROJEÇÕES (só extração)                                          │
├─────────────────────────────────────────────────────────────────────┤
│  📚 Legado (14.9M docs):  {h_legado:>10,.0f}h ({h_legado/24:,.0f} dias)              │
│  📅 Mensal (181K docs):   {h_mensal:>10.1f}h                              │
└─────────────────────────────────────────────────────────────────────┘

📂 Arquivos salvos em: {DIR_TEXTOS}
""")
    
    # Salvar resultado detalhado
    resultado = {
        "data": datetime.now().isoformat(),
        "etapa": "extracao",
        "config": {
            "max_chars": MAX_CHARS,
            "amostra": AMOSTRA,
            "dir_pdfs": str(DIR_PDFS),
            "dir_textos": str(DIR_TEXTOS)
        },
        "processamento": {
            "total_pdfs": len(pdfs),
            "cache": len(cache),
            "novos_processados": len(novos),
            "extraidos": extraidos,
            "erros": erros
        },
        "metricas": {
            "tempo_total_s": round(t_total, 2),
            "tempo_carga_pymupdf_s": round(t_load, 2),
            "tempo_medio_por_doc_s": round(segundos_por_doc, 3),
            "docs_hora": round(docs_hora, 0)
        },
        "dados": {
            "tamanho_pdfs_mb": round(total_bytes / (1024 * 1024), 2),
            "caracteres_total": total_chars,
            "tokens_estimados": total_tokens,
            "media_tokens_por_doc": round(total_tokens / extraidos, 0) if extraidos > 0 else 0
        },
        "projecoes": {
            "legado_docs": TOTAL_LEGADO,
            "legado_horas": round(h_legado, 0),
            "legado_dias": round(h_legado / 24, 0),
            "mensal_docs": TOTAL_MENSAL,
            "mensal_horas": round(h_mensal, 1)
        },
        "diretorio_saida": str(DIR_TEXTOS)
    }
    
    arq = DIR_RESULTADOS / f"extracao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(arq, "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Métricas salvas em: {arq}")


if __name__ == "__main__":
    main()