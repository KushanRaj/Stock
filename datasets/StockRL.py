from torch.utils.data import Dataset
import numpy as np
import os 
from glob import glob
import torch
import pandas as pd


class Stocks(Dataset):

    
    def __init__(self, config):
        super().__init__()
        self.root = config["root"]
        self.channel = config["channels"]
        self._load_paths()

    def _load_paths(self):

        self.data = torch.tensor(np.array(pd.read_csv(self.root,index_col=0))).float()
        self.IMG_SIZE = np.sqrt(self.data[0].size(0)).astype(int)

    def __getitem__(self,ind):
        image = self.data[ind:ind+self.channel].view(1,self.channel,self.IMG_SIZE,self.IMG_SIZE)
        price = self.data[ind+self.channel][1]
        close = self.data[ind+self.channel][0]

        return (image,
               float(price),
               float(close))

    def __len__(self):

        return len(self.data) - self.channel




    