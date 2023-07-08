#train_nn.py
import threading
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import  ArtistAnimation
import time

from utils import*
from reversi_nn import OthelloAgent8
from reversi_az import AlphaZeroAgent

def train():
    agent_az=AlphaZeroAgent()
    agent_az.load_checkpoint()
    
    for i in range(3):
        agent_az.sim_games(5)
        agent_az.save_wins(agent_az.win_probs_this_time)
        batchs=32
        for j in range(30):
            agent_az.load_wins(filename='win_probs_nn.pkl',load_len=batchs)
            samples=agent_az.sample_from_whole(batchs)
            boards=[]
            answers=[]
            for board,answer in samples:
                board=torch.from_numpy(np.array(ex_board(board),dtype=np.float32))
                answer=torch.from_numpy(np.array(answer,dtype=np.float32))
                boards.append(board)
                answers.append(answer)
            boards=torch.stack(boards,dim=0).view(batchs,1,10,10)
            answers=torch.stack(answers,dim=0)
            agent_az.train_dir(boards,answers)
        print(f'{i}-th loop in train_nn')



def computer_0(board,iro):
    while True:
        rx=random.randint(0,7)
        ry=random.randint(0,7)
        if kaeseru(rx,ry,board,iro=iro):
            return rx,ry


def make_frame(board,win_probs,iro):
    frame_object=[]
    for i in range(8):
        for j in range(8):
            x=[i,i,i+1,i+1]
            y=[7-j,8-j,8-j,7-j]
            #print(plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*8+i],0,1)))
            if iro==BLACK: frame_object.extend(plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*8+i],0,1)))
            if board[j][i]==BLACK:
                frame_object.append(patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='b', ec='b'))
            elif board[j][i]==WHITE:
                frame_object.append(patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='w', ec='b'))
    return frame_object #pip install matplotlib==3.5.1

def vs_rand_demo():
    agent_az_demo=AlphaZeroAgent()
    agent_az_demo.load_checkpoint()
    plt.title('AlphaZero vs Random')
    plt.xlim(0, 8)
    plt.ylim(0, 8)

    for i in range(1):
        print(i)
        frames=[]
        board=ban_syokika(8,8)
        iro=BLACK
        steps=0
        while True:
            plt.cla()
            if not uteru_masu(board,BLACK) and not uteru_masu(board,WHITE):
                break
            if iro==BLACK:
                board_nn=board_list2tensor(ex_board(to_black(deepcopy(board),iro)))
                x,y=agent_az_demo.select_action(board)
                win_probs=agent_az_demo.predict(board_nn)[0].tolist()
            else:
                x,y=computer_0(board,iro)
            board=ishi_utsu(x,y,board,iro)
            #frames.append(make_frame(board,win_probs,iro))
            for i in range(8):
                for j in range(8):
                    x=[i,i,i+1,i+1]
                    y=[7-j,8-j,8-j,7-j]
                    #print(plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*8+i],0,1)))
                    if iro==BLACK: plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*8+i],0,1))
                    if board[j][i]==BLACK:
                        c=patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='b', ec='b')
                        ax.add_patch(c)
                    elif board[j][i]==WHITE:
                        c=patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='w', ec='b')
                        ax.add_patch(c)
            steps+=1
            iro=3-iro
            time.sleep(0.5)
            plt.pause(0.1)
            

fig= plt.figure(figsize=(3,3))
ax=plt.axes()
vs_rand_demo()