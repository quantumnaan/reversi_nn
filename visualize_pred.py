#visualize_pred.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as patches
from computer import computer_MC
import pickle
from reversi_az import AlphaZeroAgent,ex_board,board_list2tensor,mask_kaeseru
from copy import deepcopy
BLACK=1
WHITE=2
#agent_az=AlphaZeroAgent()
#agent_az.load_checkpoint()
agent_mc=computer_MC(6,6)
agent_mc.load_wins(filename="win_probs_nn66.pkl",load_len=10)
results=agent_mc.sample(1)
# agent_az.sim_games(1)
# results=agent_az.sample_this_time(5)

# with open('tmp_save.pkl','wb') as f:
#     pickle.dump(results,f)

# with open('win_probs.pkl','rb') as f:
#     results=pickle.load(f)

boards=[i[0] for i in results[:1]]
#win_probs=[]#[i[1] for i in results]
win_probs_mc=[]
win_probs=[results[0][1]]
for i in range(len(results[:1])):
    win_probs_mc.append(agent_mc.predict(BLACK,50,boards[i]))
    board=board_list2tensor(ex_board(boards[i]))
    #tmp_win_prob=agent_az.predict(board)[0].tolist()
    #print(len(tmp_win_prob))
    #win_probs.append(tmp_win_prob)


print('nn:\n',win_probs[0])
print('mc:\n',win_probs_mc[0])

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title('Monte Carlo')
ax2.set_title('AlphaZero')
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 6)
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 6)

#print(win_probs)

def make_ax(ax,win_probs,board):
    for i in range(6):
        for j in range(6):
            x=[i,i,i+1,i+1]
            y=[5-j,6-j,6-j,5-j]
            if win_probs[j*6+i]>0: 
                ax.fill(x, y, color="green", alpha=np.clip(win_probs[j*6+i],0,1))
            else: 
                ax.fill(x, y, color="red", alpha=np.clip(-win_probs[j*6+i],0,1))

            if board[j][i]==BLACK:
                c1 = patches.Circle(xy=(i+0.5, 5.5-j), radius=0.2, fc='b', ec='b')
            elif board[j][i]==WHITE:
                c1 = patches.Circle(xy=(i+0.5, 5.5-j), radius=0.2, fc='w', ec='b')
            if board[j][i]!=0:
                ax.add_patch(c1)
    #return ax

make_ax(ax1,win_probs_mc[0],boards[0])
make_ax(ax2,win_probs[0],boards[0])
plt.show()