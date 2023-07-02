import random
import pickle
import os
import copy

BLACK=1
WHITE=2

class computer_MC:
    def __init__(self,wx,wy):
        self.wx=wx
        self.wy=wy
        self.board=[[0]*wx for _ in range(wy)]
        self.back=[[0]*wx for _ in range(wy)]
        self.win_probs=[]
        self.win_probs_thistime=[]

    def kaeseru(self,x,y,iro):
        if self.board[y][x]!=0:
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
                    if self.board[sy][sx]==0:
                        break
                    if self.board[sy][sx]==3-iro:
                        k+=1
                    if self.board[sy][sx]==iro:
                        total+=k
                        break
        return total

    def uteru_masu(self,iro):
        for y in range(8):
            for x in range(8):
                if self.kaeseru(x,y,iro)>0:
                    return True

        return False

    def ishino_kazu(self):
        b=0
        w=0
        for y in range(8):
            for x in range(8):
                if self.board[y][x]==BLACK: b+=1
                if self.board[y][x]==WHITE: w+=1

        return b,w

    def save(self):
        for y in range(8):
            for x in range(8):
                self.back[y][x] = self.board[y][x]

    def load(self):
        for y in range(8):
            for x in range(8):
                self.board[y][x] = self.back[y][x]
    
    def ishi_utsu(self,x,y,iro):
        self.board[y][x]=iro
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
                    if self.board[sy][sx]==0:
                        break
                    if self.board[sy][sx]==3-iro:
                        k+=1
                    if self.board[sy][sx]==iro:
                        for i in range(k):
                            sx-=dx
                            sy-=dy
                            self.board[sy][sx]=iro
                        break

    def uchiau(self,iro):
        while True:
            if self.uteru_masu(BLACK)==False and self.uteru_masu(WHITE)==False:
                break
            iro = 3-iro
            if self.uteru_masu(iro)==True:
                while True:
                    x = random.randint(0, 7)
                    y = random.randint(0, 7)
                    if self.kaeseru(x, y, iro)>0:
                        self.ishi_utsu(x, y, iro)
                        break

    def predict(self,iro,loops,board):
        self.board=board
        win = [0]*64
        self.save()
        for y in range(8):
            for x in range(8):
                if self.kaeseru(x, y, iro)>0:
                    #msg += "."
                    #banmen()
                    win[x+y*8] = 1
                    for i in range(loops):
                        self.ishi_utsu(x, y, iro)
                        self.uchiau(iro)
                        b, w = self.ishino_kazu()
                        if iro==BLACK and b>w:
                            win[x+y*8] += 1
                        if iro==WHITE and w>b:
                            win[x+y*8] += 1
                        self.load()
        return [w/loops for w in win]
    
    def to_black(self,board,iro):
        if iro==BLACK:
            return board
        for x in range(self.wx):
            for y in range(self.wy):
                if board[y][x]==3-iro: board[y][x]=iro
                elif board[y][x]==iro: board[y][x]=3-iro
        return board


    def action(self,iro,loops,board):
        wins=self.predict(iro,loops,copy.deepcopy(board))
        if 0 in sum(board,[]): 
            copy_b=copy.deepcopy(board)
            copy_b=self.to_black(copy_b,iro)
            self.win_probs_thistime.append((copy_b,wins))
        max_id=wins.index(max(wins))
        x=max_id%self.wx
        y=max_id//self.wy
        return x,y

    def save_wins(self,win_probs:list,folder='record',filename='win_probs.pkl'):
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

    def load_wins(self,folder='record',filename='win_probs.pkl',load_len=300):
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


