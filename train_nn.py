#train_nn.py
from reversi_nn import OthelloAgent8
from copy import deepcopy
import numpy as np

loop=50

wx=8
wy=8
BLACK=1
WHITE=2
agent_b=OthelloAgent8()
agent_w=OthelloAgent8()
board=[[0]*wx for _ in range(wy)]
board[3][4]=BLACK
board[4][3]=BLACK
board[3][3]=WHITE
board[4][4]=WHITE
back=deepcopy(board)

agent_b.load_checkpoint()
agent_w.load_checkpoint()

def kaeseru(board,x,y,iro):
    if board[y][x]!=0:
        return -1
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
    return total

def uteru_masu(board,iro):
    for y in range(wx):
        for x in range(wy):
            if kaeseru(board,x,y,iro)>0:
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

def save():
    for y in range(8):
        for x in range(8):
            back[y][x] = board[y][x]

def load():
    for y in range(8):
        for x in range(8):
            board[y][x] = back[y][x]

def ishi_utsu(board,x,y,iro):
    board[y][x]=iro
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
                    for i in range(k):
                        sx-=dx
                        sy-=dy
                        board[sy][sx]=iro
                    break
    return board

def to_black(board,iro):
    if iro==BLACK:
        return board
    new_board=[[0]*wx for _ in range(wy)]
    for x in range(wx):
        for y in range(wy):
            if board[y][x]==3-iro: new_board[y][7-x]=iro
            elif board[y][x]==iro: new_board[y][7-x]=3-iro
    return new_board

def from_black(x,y,iro):
    if iro==BLACK:
        return x,y
    else:
        return 7-x,y

def uchiau_nn(ini_board,iro):
    board=deepcopy(ini_board)
    while True:
        if uteru_masu(board,BLACK)==False and uteru_masu(board,WHITE)==False:
            break
        if uteru_masu(board,iro)==True:
            black_board=to_black(board,iro)
            x,y=agent_b.select_action(black_board,epsilon=0.1)
            x,y=map(int,from_black(x,y,iro))
            board=ishi_utsu(deepcopy(board),x,y,iro)
        iro=3-iro
    return board

def get_winprobs(board,iro,loops):
    win=[0]*64
    save()
    for y in range(wy):
        for x in range(wx):
            if kaeseru(board,x,y,iro)>0:
                for i in range(loops):
                    ishi_utsu(deepcopy(board),x,y,iro)
                    tmp_board=uchiau_nn(board,iro)
                    b,w=ishino_kazu(tmp_board)
                    if iro==BLACK and b>w:
                        win[x+y*8] += 1
                    if iro==WHITE and w>b:
                        win[x+y*8] += 1
                    load()
                    print(f'{i}th loop in ({x},{y})')
    return [i/loops for i in win]


#ini_board=[[0]*wx for _ in range(wy)]
win_prob=get_winprobs(deepcopy(board),BLACK,10)
print(np.resize(np.array(win_prob),(8,8)))