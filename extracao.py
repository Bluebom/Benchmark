import os
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Configuração
AMOSTRA = int(os.getenv("AMOSTRA", "0"))  # 0 = todos
MAX_CHARS = int(os.getenv("MAX_CHARS", "16000"))
DIR_PDFS = Path(os.getenv("DIR_PDFS", "/workspace/tce-test/data/legislacao"))
DIR_TEXTOS = Path(os.getenv("DIR_TEXTOS", "/workspace/tce-test/data/textos"))
DIR_RESULTADOS = Path(os.getenv("DIR_RESULTADOS", "/workspace/tce-test/data/resultados"))


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
    
    print("="*60)
    print("EXTRAÇÃO DE TEXTO (DOCLING)")
    print("="*60)
    
    # Listar PDFs
    pdfs = list(DIR_PDFS.rglob("*.pdf"))
    if AMOSTRA > 0:
        pdfs = pdfs[:AMOSTRA]
    
    print(f"\n📁 PDFs encontrados: {len(pdfs)}")
    print(f"📂 Saída: {DIR_TEXTOS}")
    
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
        return
    
    # Carregar Docling
    print("\n⏳ Carregando Docling...")
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    print("✅ Docling carregado!")
    
    # Extrair novos
    extraidos = 0
    erros = 0
    total_bytes = 0
    total_tokens = 0
    
    t0 = time.time()
    
    for pdf in tqdm(novos, desc="Extraindo"):
        try:
            resultado = converter.convert(str(pdf))
            texto_completo = resultado.document.export_to_markdown()
            
            doc_data = salvar_extracao(pdf, texto_completo, len(texto_completo))
            
            extraidos += 1
            total_bytes += doc_data["tamanho_bytes"]
            total_tokens += doc_data["tokens_estimados"]
            
        except Exception as e:
            salvar_erro(pdf, str(e))
            erros += 1
    
    t_total = time.time() - t0
    docs_hora = extraidos / t_total * 3600 if t_total > 0 else 0
    
    # Resumo
    print(f"""
{'='*60}
RESUMO DA EXTRAÇÃO
{'='*60}
┌─────────────────────────────────────────────────────────┐
│  PDFs processados: {len(novos)}
│  ✅ Extraídos: {extraidos}
│  ❌ Erros: {erros}
│  📦 Cache (já existiam): {len(cache)}
├─────────────────────────────────────────────────────────┤
│  ⏱️  Tempo: {t_total:.1f}s
│  📈 Throughput: {docs_hora:,.0f} docs/hora
│  💾 Tamanho: {total_bytes/(1024*1024):.2f} MB
│  🔢 Tokens: {total_tokens:,}
├─────────────────────────────────────────────────────────┤
│  📂 Arquivos salvos em: {DIR_TEXTOS}
└─────────────────────────────────────────────────────────┘
""")
    
    # Salvar resultado
    resultado = {
        "data": datetime.now().isoformat(),
        "etapa": "extracao",
        "total_pdfs": len(pdfs),
        "cache": len(cache),
        "novos_processados": len(novos),
        "extraidos": extraidos,
        "erros": erros,
        "tempo_s": round(t_total, 2),
        "docs_hora": round(docs_hora, 0),
        "tamanho_mb": round(total_bytes/(1024*1024), 2),
        "tokens": total_tokens,
        "diretorio_saida": str(DIR_TEXTOS)
    }
    
    arq = DIR_RESULTADOS / f"extracao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(arq, "w") as f:
        json.dump(resultado, f, indent=2)
    
    print(f"💾 Log salvo: {arq}")


if __name__ == "__main__":
    main()