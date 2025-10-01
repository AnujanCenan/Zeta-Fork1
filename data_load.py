import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import wfdb
from scipy.io import loadmat

class ECGDataset(Dataset):
    def __init__(self, args):
        csv_file = pd.read_csv(args.test_csv_path)
        self.dataset_name = args.dataset_name
        self.data_path = args.dataset_path

        if self.dataset_name == 'ptbxl':
            self.labels_name = list(csv_file.columns[6:])
            self.num_classes = len(self.labels_name)
            self.ecg_path = csv_file['filename_hr']
            self.labels = csv_file.iloc[:, 6:].values

        elif self.dataset_name == 'icbeb':
            self.labels_name = list(csv_file.columns[7:])
            self.num_classes = len(self.labels_name)
            self.ecg_path = csv_file['filename'].astype(str)
            self.labels = csv_file.iloc[:, 7:].values

        elif self.dataset_name == 'chapman':
            self.labels_name = list(csv_file.columns[3:])
            self.num_classes = len(self.labels_name)
            self.ecg_path = csv_file['ecg_path'].astype(str)
            self.labels = csv_file.iloc[:, 3:].values

        else:
            raise ValueError("dataset_type should be either 'ptbxl' or 'icbeb' or 'chapman")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if self.dataset_name == 'ptbxl':
            ecg_path = self.data_path + self.ecg_path[idx]
            ecg = loadmat(ecg_path + '.mat')['feats']
            ecg = ecg.T
            ecg = ecg[:, :5000]
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
            
        elif self.dataset_name == 'icbeb':
            path_add="/g"+f"{int(self.ecg_path[idx][1])+1}/"+ self.ecg_path[idx]
            ecg_path = self.data_path + path_add
            ecg = wfdb.rdsamp(ecg_path)
            ecg = ecg[0].T
            ecg = ecg[:, :2500]
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
            
            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
            
        elif self.dataset_name == 'chapman':
            ecg_path = self.data_path + self.ecg_path[idx]
            ecg = loadmat(ecg_path)['val']
            ecg = ecg.astype(np.float32)
            ecg = ecg[:, :5000]
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
            
            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()

        # switch AVL and AVF
        # In MIMIC-ECG, the lead order is I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6
        # In downstream datasets, the lead order is I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        ecg[[4, 5]] = ecg[[5, 4]]  

        return ecg, target
