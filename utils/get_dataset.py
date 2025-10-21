import os
import json
from datasets import load_dataset
import pandas as pd


def get_datasets_raw(path: str):

    ds = load_dataset('json', data_files=path) #['train'].train_test_split(test_size=0.95, seed=42)
    
    ds = ds['train']
    
    ds_splited = ds.train_test_split(test_size=0.2, seed=42)
    
    train_split = ds_splited['test'].train_test_split(test_size=0.5, seed=42)

    ds_train = ds_splited['train']
    ds_val = train_split['train']
    ds_test = train_split['test']

    print("\n" + "="*50)
    print(f"✓ Datasets carregados:")
    print(f"  - Treino: {len(ds_train)} registros")
    print(f"  - Validação: {len(ds_val)} registros")
    print(f"  - Teste: {len(ds_test)} registros")
    print("="*50 + "\n")
    
    return ds_train, ds_val, ds_test

