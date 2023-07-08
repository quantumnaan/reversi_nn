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
agent_az=AlphaZeroAgent()
agent_az.load_checkpoint()
agent_mc=computer_MC(8,8)
# agent_az.sim_games(1)
# results=agent_az.sample_this_time(5)

# with open('tmp_save.pkl','wb') as f:
#     pickle.dump(results,f)

with open('tmp_save.pkl','rb') as f:
    results=pickle.load(f)

boards=[i[0] for i in results[:3]]
win_probs=[]#[i[1] for i in results]
win_probs_mc=[]
for i in range(len(results[:3])):
    win_probs_mc.append(agent_mc.predict(BLACK,50,boards[i]))
    board=board_list2tensor(ex_board(boards[i]))
    tmp_win_prob=agent_az.predict(board)[0].tolist()
    #print(len(tmp_win_prob))
    win_probs.append(mask_kaeseru(boards[i],tmp_win_prob))
    
    # print("board:")
    # print(boards[i])
    # print("pred_nn:")
    # print(win_probs[i])
    # print("pred_mc:")
    # print(win_probs_mc)

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title('Monte Carlo')
ax2.set_title('AlphaZero')
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 8)
ax2.set_xlim(0, 8)
ax2.set_ylim(0, 8)

board_num=2#0~2
print(win_probs)

for i in range(8):
    for j in range(8):
        x=[i,i,i+1,i+1]
        y=[7-j,8-j,8-j,7-j]
        ax1.fill(x, y, color="green", alpha=np.clip(win_probs_mc[board_num][j*8+i],0,1))
        ax2.fill(x, y, color="green", alpha=np.clip(win_probs[board_num][j*8+i],0,1))
        if boards[board_num][j][i]==BLACK:
            c1 = patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='b', ec='b')
            c2 = patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='b', ec='b')
        elif boards[board_num][j][i]==WHITE:
            c1 = patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='w', ec='b')
            c2 = patches.Circle(xy=(i+0.5, 7.5-j), radius=0.2, fc='w', ec='b')
        if boards[board_num][j][i]!=0:
            ax1.add_patch(c1)
            ax2.add_patch(c2) # onaji c wo tukaunoha nazoni support sareteinairashii
print(win_probs[2])
print(win_probs_mc[2])
plt.show()