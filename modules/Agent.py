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
        self.dns1 = nn.Linear(128*self.img_size*self.img_size,512)  
        self.dns2 = nn.Linear(512,action_space)  

    def forward(self,x):
        
        x = self.conv(x)
        x = x.view(-1,128*self.img_size*self.img_size)
        x = self.dns1(x) 
        x = self.dns2(x)
            
        
        return x
               


class StockAgent():


    def __init__(self,channels,action_space,img_size,batch_size,device,lr):


        
        
        self.device = device
        self.model = Model(channels,action_space,img_size).to(device)
        self.target_model = Model(channels,action_space,img_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.batch = batch_size
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        
   
    def train_step(self,X,y):
    
        self.model.train()
        
        logit = self.model(X.to(self.device))

        loss = self.loss(y,logit)
        

        

        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()
        
        

    def train(self,memory,update,discount):

        minibatch = random.sample(memory, self.batch)

        
        current_states = torch.stack([transition[0] for transition in minibatch])
        current_qs_list = self.model(current_states.to(self.device))

        
        new_current_states = torch.stack([transition[3] for transition in minibatch])
        future_qs_list = self.target_model(new_current_states.to(self.device))

        X = []
        y = []

        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            
            if not done:
                max_future_q = torch.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            
            X.append(current_state)
            y.append(current_qs)

        self.train_step(torch.stack(X),torch.stack(y))

        if update:

            self.target_model.load_state_dict(self.model.state_dict())