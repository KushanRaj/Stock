from utils.common import read_yaml
from datasets.StockRL import Stocks 
from modules.Agent import StockAgent
from tqdm import tqdm
import numpy as np
from collections import deque
import random
import torch
from torch.utils.tensorboard import SummaryWriter



class Environment():

    def __init__(self,args):
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = read_yaml(args.config)
        self.dataset = Stocks(self.config)
        self.money = self.config['money']
        self.start_money = self.money
        self.action_space = self.config['action_space']
        self.update = self.config['update_every']
        self.channels = self.config['channels']
        self.img_size = self.dataset.IMG_SIZE

        self.agent = StockAgent(self.channels,self.action_space,
                                self.img_size,self.config['batch_size'],
                                self.device,self.config['lr'])

        self.epsilon = self.config['epsilon']
        self.episodes = self.config['episodes']
        self.min_len = self.config['min_len']
        self.memory = deque(maxlen=self.config['max_len'])
        self.ep_min = self.config['ep_min']
        self.stocks = 0 
        self.writer = SummaryWriter(log_dir='logdir')
        self.discount = self.config['discount']



    def learn(self):


        
        for i in tqdm(range(self.episodes)):
            
            money = self.money
            index = random.randint(0,len(self.dataset)-self.channels)
            done = False
            
            state,_open,close = self.dataset[index]
           
            
            rewards = 0
            while (not done) and index < len(self.dataset)-1 -self.channels:
                
                if np.random.uniform()>self.epsilon:
                    
                    action,sell = self.agent.model(state.view(1,self.channels,self.img_size,self.img_size).to(self.device))
                    action = torch.argmax(action)
                    sell = sell.item()

                else:
                    action,sell = np.random.randint(3),np.random.uniform()
                
                reward,resell = self.get_reward(action,_open,close,sell)
                rewards += reward
                
                if self.money <= 0:
                    done = True

                new_state,_open,close = self.dataset[index]
                

                self.memory.append((state,action,reward,new_state,torch.tensor(resell).view(1,),done))
                
                if len(self.memory) > self.min_len:
                    update = done & i % self.update == 0
                    self.agent.train(self.memory,update,self.discount)
                state = new_state
                index += 1

                

            
            print('\n',self.stocks, self.money-money)
            self.writer.add_scalar("Money",self.money,i)
            self.writer.add_scalar("Stocks",self.stocks,i)
            self.writer.add_scalar("reward",rewards,i)
            self.writer.add_scalar("net_worth",self.money + close*self.stocks -self.start_money,i)
            self.writer.add_scalar("randomnes",self.epsilon,i)
            self.epsilon = max(self.ep_min,self.epsilon*(1-i/self.episodes))
            
    
    def get_reward(self,action,price,value,sell):
        
        delta = self.stocks*(price - value)

        if action == 0:
            self.money += (np.round(sell*(self.stocks)))*price
            delta = self.stocks*(value - price)
            
            
            self.stocks -= np.round(sell*(self.stocks))

        if action == 1:
            if self.stocks  == 0:
                stocks = 1
            else:
                stocks= self.stocks

            self.money -= np.round(sell*(stocks))*price
            self.stocks += np.round(sell*(stocks))
            
        
        if delta < 0:
            resell = .9999
        else:
            resell = 0.0001

        
                
        return (self.money - self.start_money + self.stocks*value - 10*delta),resell



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trainer script")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    env = Environment(args)
    env.learn()
