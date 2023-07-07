#reversi_nn_vdqn.py

import os
import sys
import numpy as np
from collections import deque, namedtuple
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# parameters and the strucrure have to be adjusted thorough ttial and error,
# hints: skip connection, convolution connected layer
# batch normalization
# assume here the color of the player is black without loss of generality

BLACK=1 #player
WHITE=2 #opponent

device='cuda' if torch.cuda.is_available() else 'cpu'
args={
    'channels':128,
    'kernels':2,
    'linears':128,
    'lr':1e-4,
    'batchs':64
}

Transition=namedtuple(
    'Transition',
    'state action next_state reward'
)

class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory=deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self,batch_size):
        return ranfom.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)

class NN(nn.Module):
    def __init__(self,board_x,board_y,args):
        super(NN,self).__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1,args['channels'],args['kernels']),
            nn.ReLU(),
            nn.Conv2d(args['channels'],args['channels'],args['kernels']),
            nn.ReLU(),
            nn.Conv2d(args['channels'],args['channels'],args['kernels']),
            nn.ReLU(),
            nn.Conv2d(args['channels'],args['channels'],args['kernels']),
            nn.ReLU()
        )
        self.linear_layers=nn.Sequential(
            nn.Linear(2048,args['linears']),
            nn.ReLU(),
            nn.Linear(args['linears'],args['linears']),
            nn.ReLU(),
            nn.Linear(args['linears'],board_x*board_y)
        )

    def forward(self,x):
        x=self.conv_layers(x)
        x=x.view(x.size()[0],-1)
        #print(x.size())

        x=self.linear_layers(x)
        #print(x.size())
        return x
    
    def check_cnn_size(self,size_check):
        out=self.conv_layers(size_check)
        return out.size()

class OthelloAgent8:
    def __init__(self,board_x=8,board_y=8,args=args):#assume that iro=black
        self.agent_net=NN(board_x,board_y,args).to(device)
        self.optimizer=optim.AdamW(self.agent_net.parameters(),lr=args['lr'],amsgrad=True)
        self.loss=nn.SmoothL1Loss()
        self.memory=ReplayMemory(10000)
        self.board_x=board_x
        self.board_y=board_y
        self.board=[[0]*self.board_x for _ in range(self.board_y)]
        self.back=[[0]*self.board_x for _ in range(self.board_y)]
        self.win_probs=[]
        self.win_probs_this_time=[]

    def predict(self,state):
        return self.agent_net(state)

    def kaeseru(self,x,y,board):
        iro=BLACK
        if board[y][x]!=0:
            return False
        total=0
        for dy in range(-1,2):
            for dx in range(-1,2):
                k=0
                sx=x
                sy=y
                while True:
                    sx+=dx
                    sy+=dy
                    if sx<0 or sx>7 or sy<0 or sy>7:
                        break
                    if board[sy][sx]==0:
                        break
                    if board[sy][sx]==3-iro:
                        k+=1
                    if board[sy][sx]==iro:
                        total+=k
                        break
        return total>0

    def mask_kaeseru(self,board,prediction):
        for y in range(self.board_y):
            for x in range(self.board_x):
                if not self.kaeseru(x,y,board): prediction[y*self.board_y+x]=-1
        return prediction

    def select_action(self,board:list,epsilon=0):
        rand_val=random.random()
        if rand_val>=epsilon:
            board=np.array(copy.deepcopy(board))
            board=torch.tensor(board.astype(np.float32))
            # contiguous??
            board=board.view(1,1,self.board_x,self.board_y)
            #self.agent_net eval??
            with torch.no_grad():
                prediction=self.agent_net(board)[0]
            masked_pred=self.mask_kaeseru(board.view(self.board_x,self.board_y),prediction)
            #print(masked_pred.max(0)[1])
            action=float(masked_pred.max(0)[1])
            x=action%self.board_x
            y=action//self.board_y
        else:
            while True:
                x=random.randint(0,7)
                y=random.randint(0,7)
                if self.kaeseru(x,y,board)>0:
                    break
        return x,y
    
    def train_dir(self,states,answers):
        state_action_values=self.predict(states)
        loss=self.loss(state_action_values,answers)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.agent_net.parameters(),100)
        self.optimizer.step()

    
    def save_memory(self,folder='memory',filename='transitions'):
        filepath=os.path.join(folder,filename)


    def save_checkpoint(self,folder='checkpoint',filename='checkpoint.pth.tar'):
        filepath=os.path.join(folder,filename)
        if not os.path.exists(folder):
            print("checkpoint dir does not exist, making dir")
            os.mkdir(folder)
        else:
            print("checkpoint dir exists")
        torch.save({'state_dict':self.agent_net.state_dict()},filepath)


    def load_checkpoint(self,folder='checkpoint',filename='checkpoint_fromMC.pth.tar'):
        filepath=os.path.join(folder,filename)
        if not os.path.exists(filepath):
            print(f"no model in path{filepath}")
            return 
        map_location=None if torch.cuda.is_available() else 'cpu'
        checkpoint=torch.load(filepath,map_location=map_location)
        self.agent_net.load_state_dict(checkpoint['state_dict'])

    def save_wins(self,win_probs:list,folder='record',filename='win_probs_nn.pkl'):
        filepath=os.path.join(folder,filename)
        win_probs_up2now=[]
        try:
            with open(filepath,'rb') as f:
                win_probs_up2now=pickle.load(f)
        except:
            print(f'no such file:{filepath}')
        with open(filepath,'wb') as f:
            pickle.dump(win_probs_up2now+win_probs,f)
            print('saved the games with len:',len(win_probs))
        #print(win_probs)

    def load_wins(self,folder='record',filename='win_probs_nn.pkl',load_len=300):
        filepath=os.path.join(folder,filename)
        with open(filepath,'rb') as f:
            tmp_win_probs=pickle.load(f)
            self.win_probs=tmp_win_probs[:load_len] if len(tmp_win_probs)>load_len else tmp_win_probs
            print('load games with len:',len(self.win_probs))
            #print(tmp_win_probs)

    def sample(self,samples):
        if len(self.win_probs)<=0:
            raise ValueError("the num of samples is less than 1")
        if len(self.win_probs)<=samples:
            raise ValueError("the num of samples is less than the given num:samples")
        if samples<len(self.win_probs):
            return random.sample(self.win_probs,samples)

    




if __name__=='__main__':
    from computer import computer_MC
    print("a")
    board=[]
    wx=8
    wy=8

    for _ in range(wy):
        board.append([0]*wx)
    for y in range(wy):
        for x in range(wx):
            board[y][x]=0
    board[3][4]=BLACK
    board[4][3]=BLACK
    board[3][3]=WHITE
    board[4][4]=WHITE

    mc_agent=computer_MC(wx,wy)
    agent=OthelloAgent8(board_x=wx,board_y=wy,args=args)
    agent.load_checkpoint()
    print(agent.select_action(board))

    mc_agent.load_wins(load_len=500)
    for _ in range(300):
        boards,answers=[],[]
        for board,answer in mc_agent.sample(args['batchs']):
            # in agent_mc, "board" is a list of 8x8 and "answer" is a list of 64
            board=torch.from_numpy(np.array(board,dtype=np.float32))
            answer=torch.from_numpy(np.array(answer,dtype=np.float32))
            boards.append(board)
            answers.append(answer)
        boards=torch.stack(boards,dim=0).view(args['batchs'],1,wx,wy)
        answers=torch.stack(answers,dim=0)
        #if _==0: print(board)
        
        agent.train_dir(boards,answers)

    print("board")
    print(board)
    print("mc_predict")
    print(answer)
    print("nn_predict")
    print(agent.predict(board.view(1,1,wx,wy)))

    agent.save_checkpoint()