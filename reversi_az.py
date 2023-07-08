#reversi_nn_vdqn.py

import os
import sys
import numpy as np
from collections import deque, namedtuple
import random
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# parameters and the strucrure have to be adjusted thorough ttial and error,
# hints: skip connection, convolution connected layer
# batch normalization
# what is requires_grad
# assume here the color of the player is black without loss of generality

BLACK=1 #player
WHITE=2 #opponent
c_ucb=1.0

device='cuda' if torch.cuda.is_available() else 'cpu'
args={
    'channels':128,
    'kernels':2,
    'linears':128,
    'lr_mc':1e-4,
    'lr_sim':1e-3,
    'batchs':64
}

Transition=namedtuple(
    'Transition',
    'state action next_state reward'
)



def kaeseru(x,y,board):
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
                if sx<0 or sx>len(board)-1 or sy<0 or sy>len(board[0])-1:
                    break
                if board[sy][sx]==0:
                    break
                if board[sy][sx]==3-iro:
                    k+=1
                if board[sy][sx]==iro:
                    total+=k
                    break
    return total>0

def mask_kaeseru(board,prediction):
    for y in range(len(board)):
        for x in range(len(board[0])):
            if not kaeseru(x,y,board): prediction[y*len(board)+x]=-1
    return prediction

def ban_syokika(wx,wy):
    board=[[0]*wx for _ in range(wy)]
    board[3][4]=BLACK
    board[4][3]=BLACK
    board[3][3]=WHITE
    board[4][4]=WHITE
    return board

def board_list2tensor(board:list):
    ret=np.array(deepcopy(board))
    ret=torch.tensor(ret.astype(np.float32))
    # contiguous??
    ret=ret.view(1,1,ret.size()[0],ret.size()[0])
    return ret

def board_tensor2list(board:torch.tensor,board_x,board_y):
    board.view(board_x,board_y)
    return board.tolist()


def ishi_utsu(x,y,board,iro):
    board[y][x]=iro
    for dy in range(-1,2):
        for dx in range(-1,2):
            k=0
            sx=x
            sy=y
            while True:#ERROR here: infinetely loops with k=0,(dx,dy)=(0,0)
                sx+=dx
                sy+=dy
                if sx<0 or sx>len(board[0]) or sy<0 or sy>len(board):
                    break
                if board[sy][sx]==0:
                    break
                if board[sy][sx]==3-iro:
                    k+=1
                if board[sy][sx]==iro:
                    for i in range(k):
                        sx-=dx
                        sy-=dy
                        board[sy][sx]=iro
                    break
    return board

def uteru_masu(board):#iro=BLACK
    for y in range(8):
        for x in range(8):
            if kaeseru(x,y,board)>0:
                return True

    return False

def ishino_kazu(board):
    b=0
    w=0
    for y in range(8):
        for x in range(8):
            if board[y][x]==BLACK: b+=1
            if board[y][x]==WHITE: w+=1

    return b,w

def to_black(board,iro):
    ret=deepcopy(board)
    if iro==BLACK:
        return ret
    for x in range(len(board)):
        for y in range(len(board[0])):
            if ret[y][x]==3-iro: ret[y][x]=iro
            elif ret[y][x]==iro: ret[y][x]=3-iro
    return ret

def to_original(board,iro):
    return to_black(board,iro)

def ex_board(board):
    ret=deepcopy(board)
    sy=len(ret)
    sx=len(ret[0])
    ret=[[-10]*sx]+ret+[[-10]*sx]
    for i in range(len(ret)):
        ret[i]=[-10]+ret[i]+[-10]
    return ret



class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory=deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self,batch_size):
        return ranfom.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)

class NN(nn.Module):# input:(batches,1,board_x+2,board_y+2)->output:(board_x*board_y)
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
            nn.Linear(4608,args['linears']),
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
        #TODO nn model is yet to be implemented with resnet,attention
        return x
    
    def check_cnn_size(self,size_check):
        out=self.conv_layers(size_check)
        return out.size()

class Node:
    def __init__(self,board,iro,max_d=5,prev_action=None):
        self.board=board
        self.children=[]
        self.win_probs=[]
        self.visit_time=0
        self.is_expanded=False
        self.max_d=max_d
        self.prev_action=prev_action
        self.iro=iro
        self.fin=False
    
    def expand(self):
        if not self.is_expanded and self.max_d>0:
            if uteru_masu:
                for y in range(len(self.board)):
                    for x in range(len(self.board[0])):
                        if kaeseru(x,y,self.board)>0:
                            tmp_board=deepcopy(self.board)
                            #tmp_board=to_black(board,iro)
                            tmp_board=ishi_utsu(x,y,tmp_board,iro=3-self.iro)
                            child_node=Node(tmp_board,3-self.iro,max_d=self.max_d-1,prev_action=y*len(tmp_board)+x)
                            self.children.append(child_node)
                self.is_expanded=True
            else:
                self.fin=True
    
    #TODO is there any method necessary regarding node? 

    

class AlphaZeroAgent:
    def __init__(self,board_x=8,board_y=8,args=args):#assume that iro=black
        self.agent_net=NN(board_x,board_y,args).to(device)
        self.optimizer=optim.AdamW(self.agent_net.parameters(),lr=args['lr_mc'],amsgrad=True)
        self.optimizer=optim.AdamW(self.agent_net.parameters(),lr=args['lr_sim'],amsgrad=True)
        self.loss=nn.SmoothL1Loss()
        self.memory=ReplayMemory(10000)
        self.board_x=board_x
        self.board_y=board_y
        self.board=[[0]*self.board_x for _ in range(self.board_y)]
        self.back=[[0]*self.board_x for _ in range(self.board_y)]
        self.win_probs=[]
        self.win_probs_this_time=[]
        self.win_boards=[]
        self.win_boards_this_time=[]


    def predict(self,state):
        return self.agent_net(state)

    def select_action(self,board:list,epsilon=0):
        rand_val=random.random()
        if rand_val>=epsilon:
            board=board_list2tensor(board)
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

    def single_mcts(self,node:Node):
        node.visit_time+=1
        win_prob=0
        win_index=-1#must be updated
        nn_board=board_list2tensor(ex_board(to_black(node.board,node.iro)))
        ref_win_probs=mask_kaeseru(node.board,torch.flatten(self.predict(nn_board)))
        #print(ref_win_probs)
        if node.visit_time>5 and not node.is_expanded: node.expand()
        if node.is_expanded:
            ucbs=[]
            for child in node.children:
                ucb=float(ref_win_probs[child.prev_action]) \
                        + c_ucb*np.sqrt(node.visit_time)/(1+child.visit_time)
                ucbs.append(ucb)
            if len(ucbs)==0:print(node.board)#TODO sometimes len(ubcs) becomdes zero
            win_prob,win_index=self.single_mcts(node.children[ucbs.index(max(ucbs))])
        elif node.fin:
            b,w=ishino_kazu(node.board)
            win_prob=0.9 if b>w else 1.-0.9 #TODO because 1 is too high..?
            win_index=node.prev_action
        else:
            win_prob,win_index=torch.max(ref_win_probs,dim=0)
            win_prob,win_index=float(win_prob),int(win_index)

        return 1.-win_prob,win_index

    
    def mcts(self,parent_node:Node,loops=100):
        parent_node.expand()
        win_probs=[0]*(len(parent_node.board)*len(parent_node.board[0]))
        for i in range(loops):
            win_prob,win_index=self.single_mcts(parent_node)
            try:
                if win_probs[win_index]<win_prob: win_probs[win_index]=win_prob
            except:
                print(win_probs,win_index,i)
                raise
        
        action=win_probs.index(max(win_probs))
        #print(win_probs)
        x=action%self.board_x
        y=action//self.board_y
        if not kaeseru(x,y,parent_node.board): raise
        return (x,y), win_probs

    def sim_games(self,loops):
        for _ in range(loops):
            board=ban_syokika(self.board_x,self.board_y)
            iro=BLACK
            steps=0
            boards=[]
            while True:
                steps+=1
                board=to_black(board,iro)
                if not uteru_masu(board): 
                    board=to_original(board,iro)
                    break
                parent_node=Node(board,iro)
                action,win_probs=self.mcts(parent_node)
                self.win_probs_this_time.append((ex_board(board),win_probs))
                board=ishi_utsu(action[0],action[1],board,iro=BLACK)
                boards.append(board)
                board=to_original(board,iro)
                iro=3-iro
                if steps%10==0: print(board)
            
            b,w=ishino_kazu(board)
            b_win=b>w
            for i in range(steps):
                win= 1-i%2 if b_win else i%2
                self.win_boards_this_time.append((board,win))
            print(f'is the number of win_rec the same as b_win?:',len(win_rec)==len(b_win))
            




    
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
    agent_az=AlphaZeroAgent()
    agent_az.sim_games(100)