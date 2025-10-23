

# Sentence Embedding Training

Este projeto realiza o treinamento e avaliação de modelos de sentence embedding.

## Pré-requisitos

- Instale o gerenciador de dependências [uv](https://github.com/astral-sh/uv):
	```bash
	pip install uv
	```

## Instalação das dependências

Sincronize as dependências do projeto:
```bash
uv sync
```

## Configuração

- Configure os parâmetros desejados nos arquivos de configuração (ex: `pyproject.toml`, `main.py`).

## Execução

Execute o treinamento ou processamento principal:
```bash
uv run main.py
```

Ou para processar o resumo:
```bash
uv run process_resume_json.py
```

# Input esperado

json com instancias:
{
    anchor:"texto ancora ou de referencia"
    positive:"texto semelhante que deseja alinhar o modelo para aproximar os pontos no espaço"
}

## Estrutura do Projeto

- `main.py` — Script principal de treinamento
- `process_raw_json.py` — Processa dados brutos
- `process_resume_json.py` — Processa resumos para treinamento
- `requirements.txt` — Dependências do projeto
- `models/` — Modelos treinados e checkpoints
- `utils/` — Funções utilitárias

## Observações

- Certifique-se de que os arquivos de dados estejam presentes na raiz do projeto.
- Para dúvidas ou sugestões, abra uma issue.