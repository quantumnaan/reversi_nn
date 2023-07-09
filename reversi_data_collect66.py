import tkinter
import tkinter.messagebox
import random
from computer import computer_MC

FS=("Times New Roman",30)
FL=("Times New Roman",80)
BLACK=1
WHITE=2
mx=0
my=0
mc=0
proc=0
turn=0
msg=''
space=0
color=[0]*2
who=["you","computer"]
board=[]
back=[]


for y in range(6):
    board.append([0]*6)
for y in range(6):
    back.append([0]*6)


agent_mc1=computer_MC(6,6)
agent_mc2=computer_MC(6,6)

def click(e):
    global mx,my,mc
    mc=1
    mx=int(e.x/80)
    my=int(e.y/80)
    if mx>5: mx=5
    if my>5: my=5
    # if board[my][mx]==0:
    #     ishi_utsu(mx,my,BLACK)
    # banmen()

def banmen():
    cvs.delete("all")
    cvs.create_text(320,670,text=msg,fill="silver",font=FS)
    for y in range(6):
        for x in range(6):
            X=x*80
            Y=y*80
            cvs.create_rectangle(X,Y,X+80,Y+80,outline='black')
            if board[y][x]==BLACK:
                cvs.create_oval(X+10,Y+10,X+70,Y+70,fill='black',width=0)
            if board[y][x]==WHITE:
                cvs.create_oval(X+10,Y+10,X+70,Y+70,fill='white',width=0)
    cvs.update()

def ban_syokika():
    global space
    space=60
    for y in range(6):
        for x in range(6):
            board[y][x]=0
    board[2][3]=BLACK
    board[3][2]=BLACK
    board[2][2]=WHITE
    board[3][3]=WHITE

def ishi_utsu(x,y,iro):
    board[y][x]=iro
    for dy in range(-1,2):
        for dx in range(-1,2):
            k=0
            sx=x
            sy=y
            while True:
                sx+=dx
                sy+=dy
                if sx<0 or sx>5 or sy<0 or sy>5:
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

def kaeseru(x,y,iro):
    if board[y][x]>0:
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
                if sx<0 or sx>5 or sy<0 or sy>5:
                    break
                if board[sy][sx]==0:
                    break
                if board[sy][sx]==3-iro:
                    k+=1
                if board[sy][sx]==iro:
                    total+=k
                    break
    return total

def uteru_masu(iro):
    for y in range(6):
        for x in range(6):
            if kaeseru(x,y,iro)>0:
                return True

    return False

def ishino_kazu():
    b=0
    w=0
    for y in range(6):
        for x in range(6):
            if board[y][x]==BLACK: b+=1
            if board[y][x]==WHITE: w+=1

    return b,w

def computer_0(iro):
    while True:
        rx=random.randint(0,5)
        ry=random.randint(0,5)
        if kaeseru(rx,ry,iro)>0:
            return rx,ry

def save():
    for y in range(6):
        for x in range(6):
            back[y][x] = board[y][x]

def load():
    for y in range(6):
        for x in range(6):
            board[y][x] = back[y][x]

def valid_masu(board,iro):
    ans=[]
    for y in range(len(board)):
        for x in range(len(board[0])):
            if kaeseru(x,y,board,iro=iro)>0: ans.append((x,y))
    return ans

def uchiau(iro):
    while True:
        if uteru_masu(BLACK)==False and uteru_masu(WHITE)==False:
            break
        iro = 3-iro
        if uteru_masu(iro)==True:
            while True:
                x = random.randint(0, 5)
                y = random.randint(0, 5)
                if kaeseru(x, y, iro)>0:
                    ishi_utsu(x, y, iro)
                    break

def computer_2(iro, loops):
    global msg
    win = [0]*36
    save()
    for y in range(6):
        for x in range(6):
            if kaeseru(x, y, iro)>0:
                msg += "."
                banmen()
                win[x+y*6] = 1
                for i in range(loops):
                    ishi_utsu(x, y, iro)
                    uchiau(iro)
                    b, w = ishino_kazu()
                    if iro==BLACK and b>w:
                        win[x+y*6] += 1
                    if iro==WHITE and w>b:
                        win[x+y*6] += 1
                    load()
    m = 0
    n = 0
    for i in range(36):
        if win[i]>m:
            m = win[i]
            n = i
    x = n%6
    y = int(n/6)
    return x, y

def computer_2_predict(iro, loops):
   #global msg
    win = [0]*36
    save()
    for y in range(6):
        for x in range(6):
            if kaeseru(x, y, iro)>0:
                #msg += "."
                #banmen()
                win[x+y*6] = 1
                for i in range(loops):
                    ishi_utsu(x, y, iro)
                    uchiau(iro)
                    b, w = ishino_kazu()
                    if iro==BLACK and b>w:
                        win[x+y*6] += 1
                    if iro==WHITE and w>b:
                        win[x+y*6] += 1
                    load()
    return win/loops


def main():
    global mc,proc,turn,msg,space
    banmen()
    if proc==0:
        msg="which do you choose, BALCK or WHITE"
        cvs.create_text(320,200,text="Reversi",fill="gold",font=FL)
        cvs.create_text(160,440,text="BLACK",fill="lime",font=FS)
        cvs.create_text(440,440,text="WHITE",fill="lime",font=FS)
        if mc==1:
            mc=0
            if(mx==1 or mx==2) and my==5:
                ban_syokika()
                color[0]=BLACK
                color[1]=WHITE
                turn=0
                proc=1
            if (mx==5 or mx==6) and  my==5:
                ban_syokika()
                color[0]=WHITE
                color[1]=BLACK
                turn=1
                proc=1
    elif proc==1:
        msg="your turn"
        if turn==1:
            msg="cp thinking..( ｰ̀ωｰ́ ).｡oஇ"
        proc=2
    elif proc==2:
        if turn==0:
            #cx,cy=agent_nn.select_action(board)#computer_2(color[turn],200)
            cx,cy=agent_mc1.action(color[turn],70,board)
            ishi_utsu(int(cx),int(cy),color[turn])
            space-=1
            proc=3
        else: 
            #cx,cy=agent_nn.select_action(board)#computer_2(color[turn],200)
            cx,cy=agent_mc2.action(color[turn],70,board)
            ishi_utsu(int(cx),int(cy),color[turn])
            space-=1
            proc=3
    elif proc==3:
        msg=""
        turn=1-turn
        proc=4
    elif proc==4:
        if space==0:
            proc=5
        elif uteru_masu(BLACK)==False and uteru_masu(WHITE)==False:
            tkinter.messagebox.showinfo("","end of the game")
            proc=5
        elif uteru_masu(color[turn])==False:
            tkinter.messagebox.showinfo("",who[turn]+"have no cell to put, pass the turn")
            proc=3
        else:
            proc=1
    elif proc==5:
        b,w=ishino_kazu()
        cp_win=(color[1]==BLACK and b>w) or (color[1]==WHITE and w>b)
        tkinter.messagebox.showinfo("end",f"black:{b},white:{w}")
        if (color[0]==BLACK and b>w) or (color[0]==WHITE and w>b):
            tkinter.messagebox.showinfo("","you win!")
        elif (color[1]==BLACK and b>w) or (color[1]==WHITE and w>b):
            tkinter.messagebox.showinfo("","comuputer win")
        else:
            tkinter.messagebox.show.info("","draw")
        proc=0
    root.after(100,main)
root=tkinter.Tk()
root.title("reversi")
root.resizable(False,False)
root.bind("<Button>",click)
cvs=tkinter.Canvas(width=640,height=700,bg="green")
cvs.pack()
root.after(100,main)
root.mainloop()
agent_mc1.save_wins(agent_mc1.win_probs_thistime,filename='win_probs_mc.pkl')
agent_mc2.save_wins(agent_mc2.win_probs_thistime,filename='win_probs_mc.pkl')