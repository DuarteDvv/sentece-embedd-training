from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss, GISTEmbedLoss
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
import torch

from utils.get_dataset import get_datasets_raw
from utils.evaluator import create_ir_evaluator

PATH = "raw_.json"

def print_results(results: dict):
    print("\nResultados da avaliação:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


def main():
    print("\n" + "="*50)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ CUDA disponível! Usando GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memória total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = 'cpu'
        print("⚠ CUDA não disponível. Usando CPU.")
    print("="*50 + "\n")

    model = SentenceTransformer(
        'PORTULAN/serafim-100m-portuguese-pt-sentence-encoder',
        device=device
    )

    guide = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        device=device
    )


    ds_train, ds_val, ds_test = get_datasets_raw(PATH)

    loss = GISTEmbedLoss(model=model, guide=guide)

    print("\n" + "="*50)
    print("Avaliando modelo no conjunto de teste pré treino...")
    print("="*50 + "\n")
    test_evaluator = create_ir_evaluator(ds_test, name="test")
    test_results = test_evaluator(model)
    
    print_results(test_results)

    val_evaluator = create_ir_evaluator(ds_val, name="val")

    args = SentenceTransformerTrainingArguments(
        output_dir="models/v0/", # diretório para salvar o modelo e checkpoints
        num_train_epochs=5, # número de épocas de treinamento
        per_device_train_batch_size=16, # tamanho do batch de treinamento por GPU/CPU
        per_device_eval_batch_size=16, # tamanho do batch de avaliação por GPU/CPU
        warmup_ratio=0.1, # proporção de passos de aquecimento para o agendador de taxa de aprendizado
        fp16=True,  # usar precisão mista (float16) se suportado
        bf16=False,  # usar precisão mista (bfloat16) se suportado
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # evitar duplicatas no batch

        # Reguladores
        weight_decay=0.01,  # taxa de decaimento de peso para o otimizador
        learning_rate=2e-5,  # taxa de aprendizado inicial
        max_grad_norm=1.0,  # clipping de gradiente para evitar explosão

        # Parâmetros opcionais de rastreamento/depuração: 
        eval_strategy="steps",
        eval_steps=500, # avaliar a cada 500 passos
        save_strategy="steps",
        save_steps=1000, # salvar o modelo a cada 1000 passos
        save_total_limit=2, # manter apenas os 2 últimos checkpoints
        logging_steps=100, # registrar métricas a cada 100 passos
        # run_name="mpnet-base-all-nli-triplet",  # Usado no W&B se `wandb` estiver instalado
    )



    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        loss=loss,
        evaluator=val_evaluator,
    )

    print("\n" + "="*50)
    print("Iniciando treinamento...")
    print("="*50 + "\n")

    trainer.train()

    print("\n" + "="*50)
    print("Avaliando modelo no conjunto de teste pós treino...")
    print("="*50 + "\n")
    test_results = create_ir_evaluator(ds_test, name="test")(model)

    print_results(test_results)

    model.save_pretrained("models/v0/final")




if __name__ == "__main__":  
    main()