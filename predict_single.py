import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json
from transformers import T5TokenizerFast
from models.cmelt import M3AEModel
from types import SimpleNamespace
import torch.nn.functional as F
from data_load import ECGDataset

from types import SimpleNamespace


import os
from dotenv import load_dotenv

from main import (
    extract_ecg_features, 
    extract_language_features, 
    get_diseases_probs,
    load_encoders,
    categories
)

load_dotenv()
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
PTBXL_DATASET = os.getenv("PTBXL_DATASET")

DATASET_CONFIG = {
    "sub_class": {
        "dataset": "ptbxl",
        "dataset_name": "ptbxl",
        "dataset_path": PTBXL_DATASET,
        "test_csv_path": "data_split/ptbxl/sub_class/ptbxl_sub_class_test.csv"
    },
}

def single_inference(ecg_id: int):

    args = SimpleNamespace(**(DATASET_CONFIG["sub_class"]))

    df = pd.read_csv(args.test_csv_path)
    ecg_ids = set(df['ecg_id'].unique())

    if (ecg_id not in ecg_ids):
        print("ERROR: provided ecg_id does not seem to be in test csv")
        return
    
    data = df.loc[df['ecg_id'] == ecg_id]
    filename = data["filename_hr"].to_list()[0]

    print(filename)

    
    ecg_dataset = ECGDataset(args)
    all_labels = set(categories["sub_class"]) \
        .intersection(set(["CLBBB","CRBBB", "NORM", "AVB"]))
    
    json_path = "./configs/observations.json"

    with open(json_path, 'r') as file:
        converting_tool = json.load(file)
        potential_labels_C = []
        for label in all_labels:
            if label in converting_tool:
                potential_labels_C.append([
                    [obs.lower() for obs in converting_tool[label]["P"]],
                    [obs.lower() for obs in converting_tool[label]["N"]]
                ])
   
    print("Loading model encoders...")
    model, ecg_model, language_model, unimodal_ecg_pooler, unimodal_language_pooler, multi_modal_ecg_proj, multi_modal_language_proj, class_embedding = load_encoders()
    
    print("Extracting language features...")
    potential_language_features_dict = extract_language_features(language_model, potential_labels_C, unimodal_language_pooler, multi_modal_language_proj, all_labels=all_labels)
    
    train_loader = torch.utils.data.DataLoader(ecg_dataset, batch_size=800, shuffle=False, num_workers=4)





if __name__ == "__main__":
    single_inference(79)