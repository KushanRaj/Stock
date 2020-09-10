from utils.common import read_yaml
from datasets.StockRL import Stocks 
from modules.Agent import StockAgent
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter



class Environment():

    def __init__(self,args):
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = read_yaml(args.config)
        self.dataset = Stocks(self.config)
        self.start_money = self.config['money']
        
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
        self.memory = [torch.tensor([])] * 6
        self.ep_min = self.config['ep_min']
        self.maxlen = self.config['max_len']
        self.writer = SummaryWriter(log_dir='logdir')
        self.discount = self.config['discount']
        self.epoch = [0]
        


    def re__init(self,path,epoch_path=None,eps_path=None):
        self.agent.load_model(path)
        if epoch_path:
            self.epoch = np.load(epoch_path,allow_pickle=True)[0]
            self.epsilon = max(self.ep_min,self.epsilon*(1-(self.epoch)/self.episodes) )
    
    
    def learn(self):


        epoch = self.epoch
        for i in tqdm(epoch,range(self.episodes)):
            
            self.money = self.start_money
            self.stocks = 0
            index = random.randint(0,len(self.dataset)-self.channels)
            done = 0
            
            state,_open,close = self.dataset[index]
           
            stocks = 0
            rewards = 0
            while (not done) and index < len(self.dataset)-1 -self.channels:
                
                if np.random.uniform()>self.epsilon:
                    
                    action,sell = self.agent.model(state.to(self.device))
                    action = torch.argmax(action)
                    sell = sell.item()

                else:
                    action,sell = np.random.randint(3),np.random.uniform()
                
                reward,resell = self.get_reward(action,_open,close,sell)
                rewards += reward
                
                if self.money <= 0 or index >= len(self.dataset) -1 -self.channels:
                    done = 1

                new_state,_open,close = self.dataset[index+1]
                

                self.update_memory((state,torch.tensor([action]),torch.tensor([reward]),new_state,torch.tensor([[resell]]),torch.tensor([done])))
                
                if self.memory[0].size(0) > self.min_len:
                    update = done & i % self.update == 0
                    np.save('/Stock/epoch/epoch.npy',[self.epoch])
                    self.agent.train(self.memory,update,self.discount)
                state = new_state
                index += 1

                stocks += self.stocks
            
            
            
            
            self.epoch[0] = i 
            print('\n',self.stocks, rewards)
            self.writer.add_scalar("Money",self.money,i)
            self.writer.add_scalar("Stocks",np.mean(stocks),i)
            self.writer.add_scalar("reward",rewards,i)
            self.writer.add_scalar("net_worth",self.money + close*self.stocks -self.start_money,i)
            self.writer.add_scalar("randomnes",self.epsilon,i)
            self.epsilon = max(self.ep_min,self.epsilon*(1-(i)/self.episodes) )
            
    

    
    def update_memory(self,parameters):
        
        for i in range(6):
            
            self.memory[i] = torch.cat((self.memory[i],parameters[i]))

        if self.memory[0].size(0) > self.maxlen:
            self.memory = [i[1:] for i in self.memory]

    def get_reward(self,action,price,value,sell):
        
        delta = self.stocks*(price - value)

        if action == 0:
            delta = self.stocks*(value - price)
            '''
            self.money += self.stocks*price
            self.stocks = 0
            '''
            self.money += (np.round(sell*(self.stocks)))*price
            
            
            
            self.stocks -= int(np.round(sell*(self.stocks)))
            
        if action == 1:
            '''
            self.money -= price
            self.stocks += 1
            '''
            if self.stocks  == 0:
                stocks = 1
            else:
                stocks= self.stocks

            self.money -= np.round(sell*(stocks))*price
            self.stocks += int(np.round(sell*(stocks)))
            
        
        if delta < 0:
            resell = .9999
        else:
            resell = 0.0001

        
                
        return float(self.money - self.start_money + self.stocks*value - 10*delta),resell



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trainer script")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    env = Environment(args)
    env.re__init("E:/Stock/weights/model2.pth","E:/Stock/epoch/epoch.npy","E:/Stock/epoch/epsilon.npy")
    env.learn()
