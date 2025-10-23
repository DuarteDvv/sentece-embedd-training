import json
from pathlib import Path

def concatenar_descricao_solucao(resumo):
   
    descricao = resumo.get('Descrição', '')
    solucao = resumo.get('Solução', '')
    
    return f"Descrição: {descricao} Solução: {solucao}".strip()


def processar_json(arquivo_entrada, arquivo_saida):
    
    print(f"Lendo arquivo: {arquivo_entrada}")
    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    print(f"Total de registros encontrados: {len(dados)}")
    dados_processados = []
    registros_ignorados = 0
    for i, registro in enumerate(dados, 1):
        if i % 1000 == 0:
            print(f"Processando registro {i}/{len(dados)}...")
        resumo = registro.get('resumo', {})
        anchor = resumo.get('Problemática', '')
        anchor = str(anchor).strip() if anchor else ''
        positive = concatenar_descricao_solucao(resumo)
        positive = str(positive).strip() if positive else ''
        if not anchor or not positive:
            registros_ignorados += 1
            continue
        novo_registro = {
            'anchor': anchor,
            'positive': positive
        }
        dados_processados.append(novo_registro)
    print(f"Salvando arquivo: {arquivo_saida}")
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(dados_processados, f, ensure_ascii=False, indent=2)
    print(f"✅ Processamento concluído! {len(dados_processados)} registros salvos.")
    if registros_ignorados > 0:
        print(f"⚠ {registros_ignorados} registros ignorados por terem anchor ou positive vazios.")

if __name__ == '__main__':
    arquivo_entrada = 'summaries.json'
    arquivo_saida = 'resume_.json'
    if not Path(arquivo_entrada).exists():
        print(f"❌ Erro: Arquivo '{arquivo_entrada}' não encontrado!")
        exit(1)
    processar_json(arquivo_entrada, arquivo_saida)
