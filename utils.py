#utils.py
import numpy as np
import torch
from copy import deepcopy
import random

BLACK=1 #player
WHITE=2 #opponent
c_ucb=1.2
MAX_SAVE_WIN_PROB=10000

def board_cache(func):
    c_boards = {}
    def _wrapper(*args):
        if len(c_boards)>10:
            return func(*args)
        flag=False
        if len(c_boards)!=0:
            for board in c_boards.keys():
                if tuple(sum(args[1],[]))==board:
                    return c_boards[board]
        c_boards[tuple(sum(args[1],[]))] = func(*args)
        return c_boards[tuple(sum(args[1],[]))]
    return _wrapper

def kaeseru(x,y,board,iro=BLACK):
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
                if sx<0 or sx>len(board[0])-1 or sy<0 or sy>len(board)-1:
                    break
                if board[sy][sx]==0:
                    break
                if board[sy][sx]==3-iro:
                    k+=1
                if board[sy][sx]==iro:
                    total+=k
                    if total>0: return True
                    break
    return total>0

def mask_kaeseru(board,prediction_:list):
    prediction=deepcopy(prediction_)
    for y in range(len(board)):
        for x in range(len(board[0])):
            if not kaeseru(x,y,board): prediction[y*len(board)+x]=-10
    return prediction

def ban_syokika(wx,wy):
    board=[[0]*wx for _ in range(wy)]
    board[wy//2-1][wx//2]=BLACK
    board[wy//2][wx//2-1]=BLACK
    board[wy//2-1][wx//2-1]=WHITE
    board[wy//2][wx//2]=WHITE
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


def uchiau(board_,iro):#from 3-iro
    board=deepcopy(board_)
    while True:
        if not uteru_masu(board,iro=BLACK) and not uteru_masu(board,iro=WHITE):
            break
        iro=3-iro
        if uteru_masu(board,iro=iro):
            x,y=random.choice(valid_masu(board,iro))
            board=ishi_utsu(x,y,board,iro)
    return board
    

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
                if sx<0 or sx>len(board[0])-1 or sy<0 or sy>len(board)-1:
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

def uteru_masu(board,iro=BLACK):#iro=BLACK
    for y in range(len(board)):
        for x in range(len(board[0])):
            if kaeseru(x,y,board,iro):
                return True
    return False

def ishino_kazu(board):
    b=0
    w=0
    for y in range(len(board)):
        for x in range(len(board[0])):
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
    ret=[[-1]*sx]+ret+[[-1]*sx]
    for i in range(len(ret)):
        ret[i]=[-1]+ret[i]+[-1]
    return ret

def valid_masu(board,iro):
    ans=[]
    for y in range(len(board)):
        for x in range(len(board[0])):
            if kaeseru(x,y,board,iro=iro): ans.append((x,y))
    return ans