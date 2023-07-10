import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from utils import*
from computer import computer_MC


def data_collect():
    agent_mc1=computer_MC(6,6)
    agent_mc2=computer_MC(6,6)
    fig=plt.figure(figsize=(3,3))
    ax=fig.add_subplot(111)
    plt.xlim(0, 6)
    plt.ylim(0, 6)

    for loop in range(200):
        frames=[]
        board=ban_syokika(6,6)
        iro=BLACK
        steps=0
        while True:
            if not uteru_masu(board,BLACK) and not uteru_masu(board,WHITE):
                break
            if iro==BLACK:
                if uteru_masu(board,iro=iro):
                    (x,y)=agent_mc1.action(iro,100,board)
                    board=ishi_utsu(x,y,board,iro)
            else:
                if uteru_masu(board,iro=iro):
                    (x,y)=agent_mc2.action(iro,100,board)
                    board=ishi_utsu(x,y,board,iro)

            
            plt.cla()            
            ax.set_title('Monte Carlo vs Monte Carlo')
            for i in range(6):
                for j in range(6):
                    x=[i,i,i+1,i+1]
                    y=[5-j,6-j,6-j,5-j]
                    #print(plt.fill(x, y, color="green", alpha=np.clip(win_probs[j*8+i],0,1)))
                    plt.fill(x, y, color="white",edgecolor="green", alpha=1)
                    if board[j][i]==BLACK:
                        c=patches.Circle(xy=(i+0.5, 5.5-j), radius=0.2, fc='b', ec='b')
                        ax.add_patch(c)
                    elif board[j][i]==WHITE:
                        c=patches.Circle(xy=(i+0.5, 5.5-j), radius=0.2, fc='w', ec='b')
                        ax.add_patch(c)
            steps+=1
            iro=3-iro
            #time.sleep(0.5)
            plt.pause(0.05)
        
        agent_mc1.save_wins(agent_mc1.win_probs_thistime,filename='win_probs_mc.pkl')
        agent_mc2.save_wins(agent_mc2.win_probs_thistime,filename='win_probs_mc.pkl')
        agent_mc1.win_probs_thistime=[]
        agent_mc2.win_probs_thistime=[]
        if loop%5==0: print(f'{loop}-th loop done')
            

#train()
data_collect()