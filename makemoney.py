from utils.common import read_yaml
from datasets.StockRL import Stocks 
from modules.Agent import StockAgent
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import os


class Environment():

    def __init__(self,args):
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = read_yaml(args.config)
        self.dataset = Stocks(self.config)
        self.start_money = self.config['money']
        
        self.action_space = self.config['action_space']
        self.update = self.config['update_every']
        self.time_line = self.config['time_line']
        self.size = self.dataset.SIZE

        self.agent = StockAgent(self.time_line,self.action_space,
                                self.size,self.config['batch_size'],
                                self.device,self.config['lr'])

        self.epsilon = self.config['epsilon']
        self.episodes = self.config['episodes']
        self.min_len = self.config['min_len']
        self.memory = [torch.tensor([])] * 6
        self.ep_min = self.config['ep_min']
        self.maxlen = self.config['max_len']
        self.writer = SummaryWriter(log_dir='logdir')
        self.discount = self.config['discount']
        self.epoch = 0
        self.money = self.start_money
        self.stocks = 100
        


    def re__init(self,path,epoch_path=None,eps_path=None):
        self.agent.load_model(path)
        if epoch_path:
            self.epoch = np.load(epoch_path,allow_pickle=True)[0]
            self.epsilon = np.load(eps_path,allow_pickle=True)[0]
    
    
    def trade(self):


        
        
        for ind in tqdm(range(len(self.dataset))):
            
            
            
            state,_open,close = self.dataset[ind]
            
            stocks = []
            rewards = 0
            
                
                
            with torch.no_grad():
                        action,sell = self.agent.model(state.to(self.device))
                        action = torch.argmax(action)
                        sell = sell.item()

                
                
            reward,resell = self.get_reward(action,_open,close,sell)
                
            rewards += reward
            
            stocks += [self.stocks]

            
            
            self.writer.add_scalar("test/close",close,ind)
            self.writer.add_scalar("test/Money",self.money,ind)
            self.writer.add_scalar("test/Stocks",self.stocks,ind)
            self.writer.add_scalar("test/reward",reward,ind)
                
        
        self.money += self.stocks * close
        
        print(f'\nYou ended up with |{int(self.money)}| which was a diff of |{int(self.money) - self.start_money}|')
            
            
            
            
            
    



    def get_reward(self,action,price,value,sell):
        '''
        return reward for the action
        
        0 : sell
        1 : buy
        2 : hold

        delta : difference between open and close multiplied with stocks 

        if delta > 0 : then decision was wrong, and stocks sold/bought should be 0
        else decision was right and stocks sold/bought should be 100%

        reward : difference in start money and current money
        we want to earn money hence more current money means higher reward
        also a factor of delta to signify if that decision was good or not
        '''

        delta = self.stocks*(price - value)

        if action == 0:
            delta = self.stocks*(value - price)
            
            self.money += (np.round(sell*(self.stocks)))*price
            
            
            
            self.stocks -= int(np.round(sell*(self.stocks)))
            
        if action == 1:
            
            if self.stocks  == 0:
                stocks = 1
            else:
                stocks= self.stocks

            self.money -= np.round(sell*(stocks))*price
            self.stocks += int(np.round(sell*(stocks)))
            
        
        if delta < 0:
            resell = 1 - 1e-10
        else:
            resell = 1e-10

        
                
        return float(self.money - self.start_money - 10*delta),resell



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trainer script")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    env = Environment(args)
    
    env.re__init(f"/Stock/weights/{os.listdir('/Stock/weights/')[-1]}")
    env.trade()
            
