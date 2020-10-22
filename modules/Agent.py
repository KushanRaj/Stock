import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class DSCConv2d(nn.Module):

    def __init__(self,filters,out_filters,kernel_size=3,padding=0,stride=1):
        super(DSCConv2d,self).__init__()

        self.conv1 = nn.Conv2d(filters,filters, kernel_size=kernel_size,padding=padding,stride=stride,groups = filters)
        self.pointwise = nn.Conv2d(filters,out_filters,1)

    def forward(self,x):

        return self.pointwise(self.conv1(x))

class loop_conv(nn.Module):

    def __init__(self,l,filters,s):
        super(loop_conv,self).__init__()
        model = []
        
        model = [nn.Sequential(
                                    DSCConv2d(filters[i],filters[i+1], kernel_size=3,padding=1,stride=s[i]),
                                    Mish(),
                                    nn.BatchNorm2d(filters[i+1])
                                    ) for i in range (l)]
        self.model = nn.ModuleList(model)
        
    def forward(self,x):
        
        for i in self.model:
            x = i(x)
            
        return x

class Model(nn.Module):

    def __init__(self,time_line,action_space,size):

        super(Model,self).__init__()
        '''
        input -----> self-attention block ---
        --> Dense-block 1 ---> Q value
         |
         -> Dense-block 2 ---> % stocks to buy/sell   

        '''
        self.block1  =  nn.Sequential(
                                    nn.Linear(64*time_line,256),
                                    Mish(),
                                    nn.Linear(256,512),
                                    Mish(),
                                    nn.Linear(512,action_space),
                                    Mish()
                                    )
        
        self.block2  =  nn.Sequential(
                                    nn.Linear(64*time_line,256),
                                    Mish(),
                                    nn.Linear(256,1),
                                    )
        
        self.query = nn.Linear(size,64)
        self.key = nn.Linear(size,64)
        self.value = nn.Linear(size,64)
        self.time_line = time_line
        
    def forward(self,x):
        
        #self-attention block
        #----------------------
        q = self.query(x)
        v = self.value(x)
        k = self.key(x)
        score = torch.matmul(q,  k.permute(0,2,1))
        
        softscore = nn.functional.softmax(score,dim=-1)
        
        
        w = (v.repeat(1,self.time_line,1) * softscore.view(x.size(0),self.time_line*self.time_line,1)).view(x.size(0),self.time_line,self.time_line,-1).sum(1).view(x.size(0),-1)
        #-----------------------
        
        Q_val = self.block1(w)
        n = torch.sigmoid(self.block2(w))

        
        return Q_val,n
               
class StockAgent():


    def __init__(self,time_line,action_space,img_size,batch_size,device,lr):


        
        
        self.device = device
        self.model = Model(time_line,action_space,img_size).to(device)
        self.target_model = Model(time_line,action_space,img_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.batch = batch_size
        self.loss = nn.MSELoss()
        self.loss2 = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.model.eval()
   
    def train_step(self,X,y1,y2):
    
        self.model.train()
        
        q,n = self.model(X.to(self.device))

        loss1 = self.loss(q,y1) + self.loss2(n,y2)

        self.optimizer.zero_grad()
        loss1.backward()
        self.optimizer.step()
        self.model.eval()

    def load_model(self,path):

        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))
        

    def train(self,memory,update,discount):
        with torch.no_grad():
            
            torch.cuda.empty_cache()
            state,action,reward,new_states,sell,done = memory
            
            sample = torch.multinomial(torch.tensor([.1]).expand(done.size(0)),self.batch)
            
            current_qs_list,current_sell = self.model(state[sample].to(self.device))
            max_future_q = []
            future_resell = []
            for new_state in new_states[sample]:

                future_qs_list,future_sell = self.target_model(new_state.to(self.device))
                
                max_future_q += [torch.max(future_qs_list,axis=1)[0].sum().view(-1,1)]                
                future_resell += [future_sell.prod(0).view(-1,1)]


            future_resell = torch.cat(future_resell)
            max_future_q = torch.cat(max_future_q)
            
            new_q = torch.zeros((self.batch,),device=self.device)
            new_sell = torch.zeros((self.batch,1),device=self.device)
            idx = torch.where(done[sample] != True)
            idx2 = torch.where(done[sample])
            #max_future_q = torch.max(future_qs_list[idx],axis=1)[0]
            
            
            new_sell[idx] = sell[idx].to(self.device) * future_resell[idx]
            new_sell[idx2] = sell[idx2].to(self.device)

            
            new_q[idx] = reward[idx].to(self.device) + discount * max_future_q[idx].view(-1)
            new_q[idx2] = reward[idx2].to(self.device)
            current_qs_list[torch.arange(self.batch),action[sample].long()] = new_q
            current_sell[torch.arange(self.batch)] = new_sell
           
        
        self.train_step(state[sample],current_qs_list,current_sell)
        
        
        if update:

            self.target_model.load_state_dict(self.model.state_dict())
            torch.save(self.model.state_dict(), '/Stock/weights/model2.pth')