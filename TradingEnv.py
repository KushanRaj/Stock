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
        


    def re__init(self,path,epoch_path=None,eps_path=None):
        self.agent.load_model(path)
        if epoch_path:
            self.epoch = np.load(epoch_path,allow_pickle=True)[0]
            self.epsilon = np.load(eps_path,allow_pickle=True)[0]
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def learn(self):


        epoch = self.epoch
        count = 0
        for i in tqdm(range(epoch,self.episodes)):
            
            self.money = self.start_money
            self.stocks = 100
            index = random.randint(0,len(self.dataset)-self.time_line)
            done = 0
            
            state,_open,close = self.dataset[index]
            start_close = close
            stocks = []
            rewards = 0
            while (not done) and index < len(self.dataset)- 9 -self.time_line:
                
                if np.random.uniform()>self.epsilon:
                    with torch.no_grad():
                        action,sell = self.agent.model(state.to(self.device))
                        action = torch.argmax(action)
                        sell = sell.item()

                else:
                    action,sell = np.random.randint(3),np.random.uniform()
                
                reward,resell = self.get_reward(action,_open,close,sell)
                
                
                if self.money <= 0 or index >= len(self.dataset) -1 -self.time_line:
                    done = 1
                    self.money += self.stocks * close
                    reward = self.money - self.start_money

                rewards += reward
                new_state,_open,close = self.dataset[index+1]
                
                new_states = torch.cat(list(map(lambda x: self.dataset[x][0],range(index+1,index + 9)))).repeat(1,1,1,1)

                
                self.update_memory((state,torch.tensor([action],requires_grad=False),torch.tensor([reward],requires_grad=False).float(),new_states,torch.tensor([[resell]],requires_grad=False),torch.tensor([done],requires_grad=False)))
                
                if self.memory[0].size(0) > self.min_len:
                    update = done & i % self.update == 0
                    np.save('/Stock/epoch/epoch.npy',[self.epoch])
                    self.agent.train(self.memory,update,self.discount)
                    torch.save(self.agent.model.state_dict(), f'/Stock/weights/model{i}.pth')
                state = new_state
                index += 1

                stocks += [self.stocks]

                self.epoch = i 
            
                self.writer.add_scalar("close",close,count)
                self.writer.add_scalar("Money",self.money,count)
                self.writer.add_scalar("Stocks",self.stocks,count)
                self.writer.add_scalar("reward",reward,count)
                
                
                count += 1
            
            print(f'\nYou ended up with |{int(self.money)}| which was a diff of |{int(self.money) - self.start_money}|')
            print('\n', rewards)
            self.epsilon = max(self.ep_min,self.epsilon*(1-(i)/self.episodes) )
            self.writer.add_scalar("randomnes",self.epsilon,i)
            
            
    

    
    def update_memory(self,parameters):
        
        '''
        update the memory buffer and pop out the oldest entry
        '''    
        self.memory = list(map(lambda x: torch.cat((x[0],x[1])),zip(self.memory,parameters)))

        if self.memory[0].size(0) > self.maxlen:
            self.memory = list(map(lambda x : x[1:],self.memory))

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

        
                
        return float((self.money - self.start_money - delta)/self.start_money),resell



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trainer script")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    env = Environment(args)
    
    #env.re__init(f"/Stock/weights/{os.listdir('/Stock/weights/')[-1]}","E:/Stock/epoch/epoch.npy","E:/Stock/epoch/epsilon.npy")
    env.learn()
