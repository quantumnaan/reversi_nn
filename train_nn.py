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

# def train():
#     agent_az=AlphaZeroAgent(board_x=6,board_y=6)
#     agent_az.load_checkpoint(filename='checkpoint_az66.pth.tar')
    
#     for i in range(5):
#         # agent_az.sim_games(5)
#         # agent_az.save_wins(agent_az.win_probs_this_time,filename='win_probs_nn66.pkl')
#         # agent_az.win_probs_this_time=[]
#         batchs=512
#         batchs=agent_az.load_wins(filename='win_probs_mc.pkl',load_len=batchs*10) //10
#         for j in range(100):
#             samples=agent_az.sample_from_whole(batchs)
#             boards=[]
#             answers=[]
#             for board,answer in samples:
#                 board=torch.from_numpy(np.array(ex_board(board),dtype=np.float32))
#                 answer=torch.from_numpy(np.array(answer,dtype=np.float32))
#                 boards.append(board)
#                 answers.append(answer)
#             boards=torch.stack(boards,dim=0).view(batchs,1,8,8)
#             answers=torch.stack(answers,dim=0)
#             agent_az.train_dir(boards,answers)
#         print(f'{i}-th loop in train_nn done')
#         agent_az.save_checkpoint(filename='checkpoint_az66.pth.tar')
#     print(board)    
#     print(answer)
#     print(agent_az.predict(board.view(1,1,8,8)))
#     fig=plt.figure(figsize=(3,3))
#     ax=fig.add_subplot(111)
#     plt.plot(answer.tolist(),label='answer(MC)')
#     plt.plot(agent_az.predict(board.view(1,1,8,8))[0].tolist(),label='predict(NN)')
#     plt.legend()
#     plt.show()

def train_okeru():
    agent_az=AlphaZeroAgent(board_x=6,board_y=6)
    agent_az.load_checkpoint(filename='checkpoint_az66_okeru.pth.tar')
    
    for i in range(1):
        agent_az.sim_games(1)
        agent_az.save_wins(agent_az.win_probs_this_time,filename='win_probs_nn66.pkl')
        agent_az.win_probs_this_time=[]
        batchs=512
        batchs=agent_az.load_wins(filename='win_probs_nn66.pkl',load_len=batchs*10) //10
        for j in range(10):
            samples=agent_az.sample_from_whole(batchs)
            boards=[]
            answers_okeru=[]
            answers=[]
            for board,answer in samples:
                board_okeru = okeru_binary(ex_board(board))
                board = torch.from_numpy(np.array(ex_board(board),dtype=np.float32)).to(device)
                answer_okeru = torch.from_numpy(np.array(board_okeru,dtype=np.float32)).to(device)
                answer = torch.from_numpy(np.array(answer,dtype=np.float32)).to(device)
                boards.append(board)
                answers_okeru.append(answer_okeru)
                answers.append(answer)
            boards = torch.stack(boards,dim=0).view(batchs,1,8,8)
            answers_okeru = torch.stack(answers_okeru,dim=0).view(batchs,-1)
            answers = torch.stack(answers,dim=0).view(batchs,-1)
            agent_az.train_dir_okeru(boards,answers_okeru)
            agent_az.train_dir(boards,answers)
        print(f'{i}-th loop in train_nn done')
    agent_az.save_checkpoint(filename='checkpoint_az66_okeru.pth.tar')

    pred,okeru=agent_az.predict(board.view(1,1,8,8))
    fig=plt.figure(figsize=(6,3))
    ax11=fig.add_subplot(121)
    ax12=fig.add_subplot(122)
    make_frame(ax11,red_board(board.tolist()),answer.flatten().tolist(),BLACK)
    make_frame(ax12,red_board(board.tolist()),pred.flatten().tolist(),BLACK)
    # make_frame(ax21,board.tolist(),answer_okeru.flatten().tolist(),BLACK)
    # make_frame(ax22,board.tolist(),okeru.flatten().tolist(),BLACK)
    print(answer.size())
    print(pred.flatten().size())
    #plt.legend()
    plt.show()


def computer_0(board,iro):
    return random.choice(valid_masu(board,iro))
    # while True:
    #     rx=random.randint(0,7)
    #     ry=random.randint(0,7)
    #     if kaeseru(rx,ry,board,iro=iro):
    #         return rx,ry


def make_frame(ax,board,win_probs,iro):
    for i in range(6):
        for j in range(6):
            x=[i,i,i+1,i+1]
            y=[5-j,6-j,6-j,5-j]
            #print(plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*8+i],0,1)))
            if iro==BLACK: 
                win_here=win_probs[j*6+i]
                if win_here>0: ax.fill(x, y, color="green", alpha=np.clip(win_probs[j*6+i],0,1))
                else: ax.fill(x, y, color="red", alpha=np.clip(-win_probs[j*6+i],0,1))
            else : ax.fill(x, y, color="white", alpha=1)
            if board[j][i]==BLACK:
                c=patches.Circle(xy=(i+0.5, 5.5-j), radius=0.2, fc='b', ec='b')
                ax.add_patch(c)
            elif board[j][i]==WHITE:
                c=patches.Circle(xy=(i+0.5, 5.5-j), radius=0.2, fc='w', ec='b')
                ax.add_patch(c)

def vs_rand_demo():
    agent_az_demo=AlphaZeroAgent(board_x=6,board_y=6)
    agent_az_demo.load_checkpoint(filename='checkpoint_az66_okeru.pth.tar')
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
                    (x,y),win_probs=agent_az_demo.select_action_mcts(board)
                    board=ishi_utsu(x,y,board,iro)
            else:
                if uteru_masu(board,iro=iro):
                    x,y=computer_0(board,iro)
                    #(x,y), _=agent_az_demo.select_action_mcts(board)
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
        #print(board)
            

train_okeru()
vs_rand_demo()