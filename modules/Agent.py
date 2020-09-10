import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class loop_conv(nn.Module):

    def __init__(self,l,filters,s):
        super(loop_conv,self).__init__()
        model = []
        for i in range(l):
            model += nn.Sequential(
                                    nn.Conv2d(filters[i],filters[i+1], kernel_size=3,padding=1,stride=s[i]),
                                    Mish(),
                                    nn.BatchNorm2d(filters[i+1])
                                    )
        self.model = nn.ModuleList(model)
        
    def forward(self,x):
        
        for i in self.model:
            x = i(x)
            
        return x

class Model(nn.Module):

    def __init__(self,channels,action_space,img_size):

        super(Model,self).__init__()
        self.conv   = loop_conv(5,[channels,32,64,128,128,128],[1,1,2,2,2])
        self.img_size = img_size//8 + 1
        self.dns1 =  nn.Sequential(
                                    nn.Linear(128*self.img_size*self.img_size,512),
                                    Mish(),
                                    nn.BatchNorm1d(512)
                                    ) 
        self.dns2 = nn.Linear(512,action_space)
        self.dns3 = nn.Sequential(
                                    nn.Linear(512,10),
                                    Mish(),
                                    nn.BatchNorm1d(10)
                                    )
        self.dns4 = nn.Linear(10,1)    

    def forward(self,x):
        
        x = self.conv(x)
        x = x.view(-1,128*self.img_size*self.img_size)
        x = self.dns1(x) 
        q = self.dns2(x)
        
        n = self.dns3(x)
        n = torch.sigmoid(self.dns4(n))
            
        
        return q,n
               


class StockAgent():


    def __init__(self,channels,action_space,img_size,batch_size,device,lr):


        
        
        self.device = device
        self.model = Model(channels,action_space,img_size).to(device)
        self.target_model = Model(channels,action_space,img_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.batch = batch_size
        self.loss = nn.MSELoss()
        self.loss2 = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.model.train(mode=False)
   
    def train_step(self,X,y1,y2):
    
        self.model.train()
        
        q,n = self.model(X.to(self.device))

        loss1 = self.loss(q,y1.to(self.device)) + self.loss2(n,y2.to(self.device))
        
        

        

        
        
        self.optimizer.zero_grad()
        loss1.backward()
        self.optimizer.step()
        self.model.train(mode=False)

    def load_model(self,path):

        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))
        
        

    def train(self,memory,update,discount):
        
        #minibatch = random.sample(memory, self.batch)
        torch.cuda.empty_cache()
        state,action,reward,new_state,sell,done = memory
        
        sample = torch.multinomial(torch.tensor([.1]).expand(done.size(0)),self.batch)
        


        #current_states = torch.stack([transition[0] for transition in minibatch])
        current_qs_list,_ = self.model(state[sample].to(self.device))

        
        #new_current_states = torch.stack([transition[3] for transition in minibatch])
        future_qs_list,_ = self.target_model(new_state[sample].to(self.device))
        
        
        
        new_q = torch.zeros((self.batch,),device=self.device)
        idx = torch.where(done[sample] != True)
        idx2 = torch.where(done[sample])
        max_future_q = torch.max(future_qs_list[idx],axis=1)[0]
        
        new_q[idx] = reward[idx].to(self.device) + discount * max_future_q
        new_q[idx2] = reward[idx2].to(self.device)
        current_qs_list[torch.arange(self.batch),action[sample].long()] = new_q

        '''

        X = []
        y1 = []
        y2 = []

        for index, (current_state, action, reward, new_current_state, sell,done) in enumerate(minibatch):

            
            if not done:
                max_future_q = torch.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            
            X.append(current_state)
            y1.append(current_qs)
            y2.append(sell)
        '''   
        
        self.train_step(state[sample],current_qs_list,sell[sample])

        if update:

            self.target_model.load_state_dict(self.model.state_dict())
            torch.save(self.model.state_dict(), '/Stock/weights/model2.pth')