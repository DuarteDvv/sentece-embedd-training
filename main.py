from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss, GISTEmbedLoss
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
import torch
import wandb

from utils.get_dataset import get_datasets_raw
from utils.evaluator import create_ir_evaluator

PATH = "raw_.json"

def print_results(results: dict):
    print("\nResultados da avaliação:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


def main():

    run = wandb.init(
       
        project="hence",
        name="v0-serafim-gistembedloss",
        # Track hyperparameters and run metadata.
        config={
            "model_name": 'PORTULAN/serafim-100m-portuguese-pt-sentence-encoder',
            "guide_model_name": 'PORTULAN/serafim-100m-portuguese-pt-sentence-encoder',
            "dataset_path": PATH,
            "loss_function": "GISTEmbedLoss",
           
        },
    )


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
        'PORTULAN/serafim-100m-portuguese-pt-sentence-encoder',
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
        per_device_train_batch_size=32, # tamanho do batch de treinamento por GPU/CPU
        per_device_eval_batch_size=32, # tamanho do batch de avaliação por GPU/CPU
        fp16=True,  # usar precisão mista (float16) se suportado
        bf16=False,  # usar precisão mista (bfloat16) se suportado
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # evitar duplicatas no batch

        # reguladores
        weight_decay=0.01,  # taxa de decaimento de peso para o otimizador
        learning_rate=2e-5,  # taxa de aprendizado inicial
        max_grad_norm=1.0,  # clipping de gradiente para evitar explosão
        gradient_accumulation_steps = 1,  # passos de acumulação de gradiente
        warmup_ratio=0.1, # proporção de passos de aquecimento para o agendador de taxa de aprendizado
        warmup_steps = 0,  # número de passos de aquecimento (se definido, sobrescreve warmup_ratio)
        lr_scheduler_type = 'linear',  # tipo de agendador de taxa de aprendizado


        # rastreamento/depuração: 
        eval_strategy="steps",
        eval_steps=250, # avaliar a cada 250 passos
        save_strategy="steps",
        save_steps=250, # salvar o modelo a cada 250 passos
        save_total_limit=3, # manter apenas os 3 últimos checkpoints
        logging_steps=10, # registrar métricas a cada 10 passos
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # métrica para determinar o melhor modelo
        greater_is_better=False,

        report_to=["wandb"],  # Relatar métricas para W&B
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

    run.finish()




if __name__ == "__main__":  
    main()  