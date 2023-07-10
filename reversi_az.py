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
import pickle

from utils import*
from computer import computer_MC

# parameters and the strucrure have to be adjusted thorough ttial and error,
# hints: skip connection, convolution connected layer
# batch normalization
# what is requires_grad
# initialization
# assume here the color of the player is black without loss of generality


device='cuda' if torch.cuda.is_available() else 'cpu'
args={
    'channels':128,
    'kernels':2,
    'linears':128,
    'lr_mc':1e-4,
    'lr_sim':1e-3,
    'batchs':32
}

Transition=namedtuple(
    'Transition',
    'state action next_state reward'
)

# not using
class SelfAttn(nn.Module):
    def __init__(self,board_x,board_y):
        super(SelfAttn,self).__init__()
        self.selfattn=nn.MultiheadAttention((board_x+2)*(board_y+2),8)
    
    def forward(self,x):
        x,_=self.selfattn(x,x,x)
        return x


# not using
class ResNet(nn.Module):# input:(batches,1,board_x+2,board_y+2)->output:(board_x*board_y)
    def __init__(self,board_x,board_y,args):
        super(ResNet,self).__init__()
        self.block1=ResBlock(1,64)
        self.linear1=nn.Linear((board_x+2)*(board_y+2),(board_x+2)*(board_y+2))
        self.linear2=nn.Linear(1024,256)
        self.linear3=nn.Linear(256,(board_x+2)*(board_y+2))
        self.linear4=nn.Linear((board_x+2)*(board_y+2),board_x*board_y)
        self.tanh1=nn.Tanh()
        self.tanh2=nn.Tanh()
        self.tanh3=nn.Tanh()

    def forward(self,x):
        init_x=x.view(x.size()[0],-1)
        init_size=x.size()
        x=x.view(x.size()[0],-1)
        x=self.linear1(x)
        x=x.view(*init_size)
        x=self.block1(x)
        x=x.view(x.size()[0],-1)
        x=self.linear2(x)
        x=self.tanh1(x)
        x=self.linear3(x)
        x+=init_x
        x=self.linear4(x)
        x=self.tanh3(x)
        return x



class AttentionBlock(nn.Module):
    def __init__(self,board_x,board_y):
        super(AttentionBlock,self).__init__()
        self.linear1=nn.Linear((board_x+2)*(board_y+2),(board_x+2)*(board_y+2))
        self.relu1=nn.ReLU()
        self.conv1=nn.Conv2d(1,128,kernel_size=3)
        self.bn1=nn.BatchNorm2d(128)
        self.conv2=nn.Conv2d(128,64,kernel_size=3)
        self.bn2=nn.BatchNorm2d(64)
        self.linear_form=nn.Linear(64*4*4,(board_x+2)*(board_y+2))
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,x):
        size_init=x.size()
        x=x.view(x.size()[0],-1)
        x=self.linear1(x)
        x=self.relu1(x)
        x_tmp=x
        x_tmp_size=x_tmp.size()
        x=x.view(*size_init)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        #print(x.size())
        x=x.view(x.size()[0],-1)
        x=self.linear_form(x)
        x=self.sigmoid(x)
        
        x=torch.mul(x,x_tmp.view(x_tmp.size()[0],-1))
        return x



class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.sigmoid=nn.Sigmoid()
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.linear_shape=nn.Linear(8*8,64*4*4)
    
    def forward(self,x):
        x_init=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.sigmoid(x)
        x=self.conv2(x)
        x=self.bn2(x)
        #print(x.size(),x_init.size())
        x_init=self.linear_shape((x_init).view(x_init.size()[0],-1))
        x+=x_init.view(x_init.size()[0],-1,4,4)
        return x


class ResNet2(nn.Module):# input:(batches,1,board_x+2,board_y+2)->output:(board_x*board_y)
    def __init__(self,board_x,board_y,args):
        super(ResNet2,self).__init__()
        self.attention=AttentionBlock(board_x,board_y)#AttentionBlock(board_x,board_y)
        self.block1=ResBlock(1,64)
        self.linear1=nn.Linear((board_x+2)*(board_y+2),(board_x+2)*(board_y+2))
        self.linear2=nn.Linear(1024,256)
        self.linear3=nn.Linear(256,(board_x+2)*(board_y+2))
        self.linear4=nn.Linear((board_x+2)*(board_y+2),board_x*board_y)
        self.tanh1=nn.Tanh()
        self.tanh2=nn.Tanh()
        self.tanh3=nn.Tanh()

    def forward(self,x):
        init_size=x.size()
        #print(init_size)
        x=x.view(x.size()[0],-1)
        init_x=x
        #print(init_x.size())
        x=self.linear1(x)
        x=x.view(*init_size)
        x=self.block1(x)
        x=x.view(x.size()[0],-1)
        x=self.linear2(x)
        x=self.tanh1(x)
        x=self.linear3(x)
        x+=init_x        
        x=x.view(*init_size)
        x=self.attention(x)
        y=x
        x=self.linear4(x)
        x=self.tanh3(x)
        return x,y



class Node:
    def __init__(self,board,iro,max_d=5,prev_action=None,parent_node=None):
        self.board=board #where the next player is BLACK
        self.children=[]
        self.parent=parent_node
        self.win_probs=[0]*(len(board)*len(board[0]))
        self.visit_time=0
        self.is_expanded=False
        self.max_d=max_d
        self.prev_action=prev_action
        self.iro=iro
        self.fin=False
    
    def expand(self):
        if (not self.is_expanded) and self.max_d>0:
            if uteru_masu(self.board):
                for y in range(len(self.board)):
                    for x in range(len(self.board[0])):
                        if kaeseru(x,y,self.board):
                            tmp_board=deepcopy(self.board)
                            tmp_board=ishi_utsu(x,y,tmp_board,iro=BLACK)
                            tmp_board=to_black(tmp_board,3-self.iro)
                            child_node=Node(tmp_board,3-self.iro,max_d=self.max_d-1,\
                                            prev_action=y*len(tmp_board)+x,parent_node=self)
                            self.children.append(child_node)
                self.is_expanded=True
            else:
                self.fin=True
    
    #TODO is there any method necessary regarding node? 

    

class AlphaZeroAgent:
    def __init__(self,board_x=6,board_y=6,args=args):#assume that iro=black
        self.agent_net=ResNet2(board_x,board_y,args).to(device)
        self.optimizer=optim.AdamW(self.agent_net.parameters(),lr=args['lr_mc'],amsgrad=True)
        self.optimizer_okeru=optim.AdamW(self.agent_net.parameters(),lr=args['lr_sim'],amsgrad=True)
        self.loss=nn.SmoothL1Loss()
        self.loss_okeru=nn.MSELoss()
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
            board_nn=deepcopy(board)
            board_nn=ex_board(board_nn)
            board_nn=board_list2tensor(board_nn)
            # contiguous??
            board_nn=board_nn.view(1,1,self.board_x+2,self.board_y+2)
            #self.agent_net eval??
            with torch.no_grad():
                prediction=self.agent_net(board_nn)[0]
            masked_pred=mask_kaeseru(board,prediction.tolist())

            action=masked_pred.index(max(masked_pred))
            x=action%self.board_x
            y=action//self.board_y
        else:
            x,y=random.choice(valid_masu(board,iro=BLACK))
        return x,y

    def select_action_mcts(self,board:list,epsilon=0):
        rand_val=random.random()
        win_probs=[0]*(len(board)*len(board[0]))
        if rand_val>=epsilon:
            current_node=Node(board,iro=BLACK)
            (x,y),win_probs=self.mcts(current_node)
        else:
            x,y=random.choice(valid_masu(board,iro=BLACK))
        return (x,y),win_probs


    def train_dir_okeru(self,boards,okerus):
        _,okeru_pred=self.predict(boards)
        loss=self.loss_okeru(okeru_pred,okerus)
        self.optimizer_okeru.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.agent_net.parameters(),100)
        self.optimizer.step()
        return okeru_pred


    def train_dir(self,states,answers):
        state_action_values,_=self.predict(states)
        loss=self.loss(state_action_values,answers)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.agent_net.parameters(),100)
        self.optimizer.step()
        return state_action_values

    @board_cache
    def simple_mcts(self,board,loops=10):
        iro=BLACK
        board_init=board
        board_here=deepcopy(board)
        win=[0]*(len(board)*len(board[0]))
        kaeshita=[False]*(len(board)*len(board[0]))
        for y in range(len(board)):
            for x in range(len(board[0])):
                if kaeseru(x,y,board_here):
                    for i in range(loops):
                        board_here=ishi_utsu(x,y,board_here,iro)
                        board_here=uchiau(board_here,iro)
                        b,w=ishino_kazu(board_here)
                        kaeshita[x+y*len(board[0])]=True
                        if b>w:
                            win[x+y*len(board[0])]+=1
                        board_here=deepcopy(board_init)
        win=[2.*w/loops-1. if k else 0 for (w,k) in zip(win,kaeshita) ]
        return win

    def single_mcts(self,node:Node):
        #return action selected by pucb and corresponding win probs in this board 
        node.visit_time+=1
        win_prob=0 
        win_index=None#must be updated

        # if sum(node.board,[]).count(0)<=4:
        #     win_prob,win_index=self.simple_mcts(node.board)
        # else:
        nn_board=board_list2tensor(ex_board(to_black(node.board,node.iro)))
        ref_win_probs=torch.flatten(self.predict(nn_board)).tolist()
        node.win_probs=ref_win_probs
        if node.visit_time>=5 and not node.is_expanded: node.expand()

        if node.is_expanded:
            ucbs=[]
            for child in node.children:
                action_prob=((ref_win_probs[child.prev_action]+1.)/2)\
                                /sum([(i+1.)/2. for i in ref_win_probs])
                ucb=(ref_win_probs[child.prev_action]+1.)/2 \
                        + c_ucb*(action_prob+0.02)*np.sqrt(np.log(node.visit_time))/(1+child.visit_time)
                ucbs.append(ucb)
            if len(ucbs)==0:print(node.board)#TODO sometimes len(ubcs) becomdes zero not when the game ends 
            win_prob , _ =self.single_mcts(node.children[ucbs.index(max(ucbs))])

            # because win_prob to the opponent is lose_prob to the player
            win_prob=-win_prob 
            win_index=node.children[ucbs.index(max(ucbs))].prev_action

            #update node win_probs to get close to the child win_probs, too heuristic! ('_;)
            lr_win_probs=0.1
            node.win_probs[win_index]+=lr_win_probs*(win_prob-node.win_probs[win_index])

        #this does not work because in case sum(node.board,[]).count(0)<=4 simple mcts is executed
        elif node.fin:
            #print('myproblem: node.fin becomes True in single mcts accidentally')
            b,w=ishino_kazu(node.board)
            win_prob=1. if b>w else -1.
            win_index=None

        # in a leaf node
        else:
            win_prob=-node.parent.win_probs[node.prev_action]
            win_index= None
            #pay attension, leaf node is refferd only from parent node, which does not use win_index
        
        return win_prob,win_index

    
    def mcts(self,parent_node:Node,loops=50):
        parent_node.expand()
        probs_size=len(parent_node.board)*len(parent_node.board[0])
        win_probs=[0]*(probs_size)
        probs_num=[0]*(probs_size)

        for i in range(loops):
            if sum(parent_node.board,[]).count(0)<=4:
                win_probs=self.simple_mcts(parent_node.board)

            # win_probs are stored and meaned in each node->no longer
            else:
                win_prob,win_index=self.single_mcts(parent_node)
                probs_num[win_index]+=1
                win_probs[win_index]+=1./(probs_num[win_index])*(win_prob-win_probs[win_index])

        masked_prob=mask_kaeseru(parent_node.board,win_probs)
        action=masked_prob.index(max(masked_prob) )
        #print(win_probs)
        x=action%self.board_x
        y=action//self.board_y
        if not kaeseru(x,y,parent_node.board): 
            raise
        return (x,y), win_probs

    def sim_games(self,loops):
        print(f'started simulation with {loops} loops')
        for i in range(loops):
            board=ban_syokika(self.board_x,self.board_y)
            iro=BLACK
            steps=0
            #boards=[]
            while True:
                steps+=1
                board=to_black(board,iro)
                if not uteru_masu(board): 
                    board=to_original(board,iro)
                    break
                parent_node=Node(board,iro)
                action,win_probs=self.mcts(parent_node)
                self.win_probs_this_time.append((deepcopy(board),win_probs))
                board=ishi_utsu(action[0],action[1],board,iro=BLACK)
                #boards.append(board)
                board=to_original(board,iro)
                iro=3-iro
            
            b,w=ishino_kazu(board)
            b_win=b>w
            for j in range(steps):
                win= 1-j%2 if b_win else j%2
                self.win_boards_this_time.append((board,win))

            if i%5==0: print(f'{i}-th loop in az')
            #print(f'is the number of win_rec the same as b_win?:',len(win_rec)==len(b_win))


    def save_checkpoint(self,folder='checkpoint',filename='checkpoint_az.pth.tar'):
        filepath=os.path.join(folder,filename)
        if not os.path.exists(folder):
            print("checkpoint dir does not exist, making dir")
            os.mkdir(folder)
        else:
            print("checkpoint dir exists")
        torch.save({'state_dict':self.agent_net.state_dict()},filepath)


    def load_checkpoint(self,folder='checkpoint',filename='checkpoint_az.pth.tar'):
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
            dump_data=(win_probs_up2now+win_probs)[-MAX_SAVE_WIN_PROB:] \
                if len(win_probs_up2now+win_probs)>MAX_SAVE_WIN_PROB else win_probs_up2now+win_probs
            pickle.dump(dump_data,f)
            print(f'saved games with len :{len(dump_data)},including new data with len: {len(win_probs)}')
        #print(win_probs)

    def load_wins(self,folder='record',filename='win_probs.pkl',load_len=300):
        filepath=os.path.join(folder,filename)
        with open(filepath,'rb') as f:
            tmp_win_probs=pickle.load(f)
            self.win_probs=tmp_win_probs[:load_len] if len(tmp_win_probs)>load_len else tmp_win_probs
            print('loaded games with len:',len(self.win_probs))
            #print(tmp_win_probs)
        return len(self.win_probs)

    def sample_from_whole(self,samples):
        if len(self.win_probs)<=0:
            raise ValueError("the num of samples is less than 1")
        if len(self.win_probs)<samples:
            raise ValueError("the num of samples is less than the given num:samples")
        if samples<=len(self.win_probs):
            return random.sample(self.win_probs,samples)

    def sample_this_time(self,samples):
        if len(self.win_probs_this_time)<=0:
            raise ValueError("the num of samples is less than 1")
        if len(self.win_probs_this_time)<=samples:
            raise ValueError("the num of samples is less than the given num:samples")
        if samples<len(self.win_probs_this_time):
            return random.sample(self.win_probs_this_time,samples)
    
if __name__=='__main__':
    from torchviz import make_dot
    from PIL import Image
    agent_az=AlphaZeroAgent(board_x=6,board_y=6)
    # img=make_dot(probs,params=dict(agent_az.agent_net.named_parameters()))
    # img.format="png"

    # img.render("model_img")

    dummy_input=torch.randn(1,1,8,8)#ダミーの入力を用意する
    input_names = [ "input"]
    output_names = [ "output" ]

    torch.onnx.export(agent_az.agent_net, dummy_input, "./image/test_model.onnx", verbose=True,input_names=input_names,output_names=output_names)

    # try:
    #     agent_az.load_checkpoint()
    # except:
    #     print('model does not match, but go on')
    # mc_agent=computer_MC(6,6)
    # mc_agent.load_wins()
    # for _ in range(100):
    #     boards,answers=[],[]
    #     for board,answer in mc_agent.sample(args['batchs']):
    #         # in agent_mc, "board" is a list of 8x8 and "answer" is a list of 64
    #         board=torch.from_numpy(np.array(ex_board(board),dtype=np.float32))
    #         answer=torch.from_numpy(np.array(answer,dtype=np.float32))
    #         boards.append(board)
    #         answers.append(answer)
    #     boards=torch.stack(boards,dim=0).view(args['batchs'],1,8,8)
    #     answers=torch.stack(answers,dim=0)
    #     state_action_values=agent_az.train_dir(boards,answers)
    #     #state_action_value=agent_az.predict(boards[0].view(1,1,6,6))


    # print("board",boards[0])
    # print('mc_pred',answers[0])
    # print('nn_pred',state_action_values[0])

    # agent_az.save_checkpoint()
