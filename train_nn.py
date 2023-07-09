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
    agent_az=AlphaZeroAgent(board_x=6,board_y=6)
    agent_az.load_checkpoint(filename='checkpoint_az.pth.tar')
    
    for i in range(1):
        #agent_az.sim_games(10)
        #agent_az.save_wins(agent_az.win_probs_this_time)
        batchs=1024
        agent_az.load_wins(filename='win_probs.pkl',load_len=batchs*10)
        for j in range(30):
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
        print(f'{loop}-th loop in train_nn')
    agent_az.save_checkpoint(filename='checkpoint_az_88.pth.tar')



def computer_0(board,iro):
    return random.choice(valid_masu(board,iro))
    # while True:
    #     rx=random.randint(0,7)
    #     ry=random.randint(0,7)
    #     if kaeseru(rx,ry,board,iro=iro):
    #         return rx,ry


# def make_frame(board,win_probs,iro):
#     frame_object=[]
#     for i in range(8):
#         for j in range(8):
#             x=[i,i,i+1,i+1]
#             y=[7-j,8-j,8-j,7-j]
#             #print(plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*8+i],0,1)))
#             if iro==BLACK: frame_object.extend(plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*8+i],0,1)))
#             if board[j][i]==BLACK:
#                 frame_object.append(patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='b', ec='b'))
#             elif board[j][i]==WHITE:
#                 frame_object.append(patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='w', ec='b'))
#     return frame_object #pip install matplotlib==3.5.1

def vs_rand_demo():
    agent_az_demo=AlphaZeroAgent(board_x=6,board_y=6)
    agent_az_demo.load_checkpoint(filename='checkpoint_az.pth.tar')
    fig=plt.figure(figsize=(3,3))
    ax=fig.add_subplot(111)
    plt.xlim(0, 6)
    plt.ylim(0, 6)

    for loop in range(1):
        frames=[]
        board=ban_syokika(6,6)
        iro=BLACK
        steps=0
        while True:
            if not uteru_masu(board,BLACK) and not uteru_masu(board,WHITE):
                break
            if iro==BLACK:
                if uteru_masu(board,iro=iro):
                    board_nn=board_list2tensor(ex_board(to_black(deepcopy(board),iro)))
                    (x,y),win_probs=agent_az_demo.select_action_mcts(board)
                    board=ishi_utsu(x,y,board,iro)
            else:
                if uteru_masu(board,iro=iro):
                    x,y=computer_0(board,iro)
                    board=ishi_utsu(x,y,board,iro)
            #frames.append(make_frame(board,win_probs,iro))
            
            plt.cla()            
            ax.set_title('AlphaZero vs Random')
            for i in range(6):
                for j in range(6):
                    x=[i,i,i+1,i+1]
                    y=[5-j,6-j,6-j,5-j]
                    #print(plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*8+i],0,1)))
                    if iro==BLACK: 
                        win_here=win_probs[j*6+i]
                        if win_here>0: plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*6+i],0,1))
                        else: plt.fill(x, y, color="red", alpha=np.clip(-win_probs[j*6+i],0,1))
                    else : plt.fill(x, y, color="white", alpha=1)
                    if board[j][i]==BLACK:
                        c=patches.Circle(xy=(i+0.5, 5.5-j), radius=0.2, fc='b', ec='b')
                        ax.add_patch(c)
                    elif board[j][i]==WHITE:
                        c=patches.Circle(xy=(i+0.5, 5.5-j), radius=0.2, fc='w', ec='b')
                        ax.add_patch(c)
            steps+=1
            iro=3-iro
            #time.sleep(0.5)
            plt.pause(0.5)
        plt.show()
        print(board)
            

#train()
vs_rand_demo()