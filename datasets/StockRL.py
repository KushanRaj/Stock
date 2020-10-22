from torch.utils.data import Dataset
import numpy as np
import os 
from glob import glob
import torch
import pandas as pd


class Stocks(Dataset):
    
    
    def __init__(self, config):
        '''
        Create a dataset containg stock data
        returns a set number of days if data as one item
        '''
        super().__init__()
        self.root = config["root"]  #find the root file for dataset
        self.time_line = config["time_line"] #no of days being taken together as on input
        self._read_paths() # read csv doc containing stock data

    def _read_paths(self):
        '''
        read the csv file
        '''
        self.data = torch.tensor(np.array(pd.read_csv(self.root,index_col=0)),requires_grad=False).float() # data is in Long() format, we need float
        self.SIZE = self.data[0].size(0)  # No. dimensions of data of each day, i.e, No. of market value indicators
    
    def __getitem__(self,ind):
        '''
        returns data for the required index
        '''
        data = self.data[ind:ind+self.time_line].view(1,self.time_line,self.SIZE) # time_line days as one item
        price = self.data[ind+self.time_line - 1][1] # open value of the final day
        close = self.data[ind+self.time_line - 1][0] # close value of the final day

        return (data,
               float(price),
               float(close))

    def __len__(self):
        '''
        returns length of the document
        '''
        return len(self.data) - self.time_line 




    