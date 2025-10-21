import json
from pathlib import Path


def concatenar_interacoes(interacoes):
    """
    Concatena todas as mensagens anonimizadas das interações em uma única string.
    
    Args:
        interacoes: Lista de dicionários com as interações
        
    Returns:
        String com todas as mensagens concatenadas com separadores
    """
    mensagens = []
    for interacao in interacoes:

        ordem = interacao.get('ordem', 0)
        autor = interacao.get('autor', 'DESCONHECIDO')

     
        anon = interacao.get('mensagem_anonimizada')
        orig = interacao.get('mensagem')

        use_msg = None
        if anon is None:
            use_msg = orig
        else:

            try:
                import math
                if isinstance(anon, float) and math.isnan(anon):
                    use_msg = orig
                else:
         
                    if isinstance(anon, str) and anon.strip().lower() == 'nan':
                        use_msg = orig
                    else:
                        use_msg = anon
            except Exception:
                
                use_msg = anon if anon is not None else orig

        if use_msg is None:
            msg_str = ''
        else:
            msg_str = str(use_msg).replace('\n', ' ').strip()

        mensagens.append(f"{autor}: {msg_str}")

    return ' '.join([m for m in mensagens if m])


def processar_json(arquivo_entrada, arquivo_saida):
    """
    Lê o arquivo JSON de entrada e cria um novo JSON com apenas
    reclamacao_anonimizada e interacoes concatenadas.
    
    Args:
        arquivo_entrada: Caminho do arquivo JSON de entrada
        arquivo_saida: Caminho do arquivo JSON de saída
    """
    print(f"Lendo arquivo: {arquivo_entrada}")
    
    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    
    print(f"Total de registros encontrados: {len(dados)}")
    
    dados_processados = []
    registros_ignorados = 0
    for i, registro in enumerate(dados, 1):
        if i % 1000 == 0:
            print(f"Processando registro {i}/{len(dados)}...")
        
        # Obter anchor com fallback para reclamacao original
        anchor = registro.get('reclamacao_anonimizada')
        if anchor is None or (isinstance(anchor, str) and anchor.strip() == ''):
            anchor = registro.get('reclamacao', '')
        
        # Garantir que anchor seja string válida
        if anchor is None:
            anchor = ''
        else:
            anchor = str(anchor).strip()
        
        # Obter positive (interações concatenadas)
        positive = concatenar_interacoes(registro.get('interacoes', []))
        if positive is None:
            positive = ''
        else:
            positive = str(positive).strip()
        
        # Ignorar registros onde anchor ou positive estão vazios
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
  
    arquivo_entrada = 'evaluation_data.json'
    arquivo_saida = 'raw_.json'
    
    if not Path(arquivo_entrada).exists():
        print(f"❌ Erro: Arquivo '{arquivo_entrada}' não encontrado!")
        exit(1)
    

    processar_json(arquivo_entrada, arquivo_saida)
