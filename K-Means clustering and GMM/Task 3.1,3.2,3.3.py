import numpy as np
from math import e,pi,sqrt
import matplotlib.pyplot as plt
u = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])
x = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])

gar = []
dist= []
xa=[]
ya=[]
co=[]
u1=[]
u2=[]
u3=[]
tot=[]


def kmeans(x,uz):
    ux=np.array([[0.0 for j in range(2)] for i in range(3)])
    yes=0
#    mean.append(uz)
    u1[:]=[]
    u2[:]=[]
    u3[:]=[]
    tot[:]=[]
    dist[:] = []
    for i in range(len(x)):
        gar=[]
        for j in range(len(uz)):
            distance = sqrt((uz[j][0]-x[i][0])**2 + (uz[j][1]-x[i][1])**2)
            gar.append(distance)
        y= gar.index(min(gar))
        if(y==0):
            col='red'
        if(y==1):
            col='green'
        if(y==2):
            col='blue'
        gar.append(col)
        gar.append(y)
        dist.append(gar)
#        print(dist)
     
    for i in range(len(dist)):
        if(dist[i][4]==0):
            u1.append(x[i])
        if(dist[i][4]==1):
            u2.append(x[i])
        if(dist[i][4]==2):
            u3.append(x[i])
            
            
    tot.append(u1)
    tot.append(u2)
    tot.append(u3)
#    sto = uz

#    print(sto) 
    for i in range(3):
        ux[i][0]=sum([x[0] for x in tot[i]])/len(tot[i])
        ux[i][1]=sum([x[1] for x in tot[i]])/len(tot[i])


    for i in range(3):
        for j in range(2):
            if(ux[i][j]==uz[i][j]):
                yes +=1
 
    return dist,uz,ux

       
        
    
    
di,yu,yuup = kmeans(x,u)

f = []
g = []
h = ['red','green','blue']
fu = []
gu = []


for i in range(len(x)):
    xa.append(x[i][0])
    ya.append(x[i][1])
    co.append(di[i][3])
for i in range(3):
    f.append(yu[i][0])
    g.append(yu[i][1])

for i in range(3):
    fu.append(yuup[i][0])
    gu.append(yuup[i][1])



    

fig,ax = plt.subplots()

ax.scatter(xa, ya, s=20, c=co, marker='^')
ax.scatter(f,g, s=40, c=h,marker = 'o')
for i,txt in enumerate(yu):
    ax.annotate(txt, (f[i],g[i]))
for i, txt in enumerate(x):
    ax.annotate(txt, (xa[i], ya[i]))   
fig.savefig('D:\\task3_iter1_a.jpg', dpi=fig.dpi)

figu,axu = plt.subplots()
axu.scatter(xa, ya, s=20, c=co, marker='^')
axu.scatter(fu,gu, s=40, c=h,marker = 'o')
for i,txt in enumerate(yuup):
    axu.annotate(txt, (fu[i],gu[i]))
for i, txt in enumerate(x):
    axu.annotate(txt, (xa[i], ya[i]))
figu.savefig('D:\\task3_iter1_b.jpg', dpi=figu.dpi)

di,yu,yuup = kmeans(x,yuup)

xa=[]
ya=[]
co=[]

f = []
g = []
h = ['red','green','blue']
fu = []
gu = []


for i in range(len(x)):
    xa.append(x[i][0])
    ya.append(x[i][1])
    co.append(di[i][3])
for i in range(3):
    f.append(yu[i][0])
    g.append(yu[i][1])
#    co.append(h[i])
for i in range(3):
    fu.append(yuup[i][0])
    gu.append(yuup[i][1])
#    co.append(h[i])


    

fig,ax = plt.subplots()

ax.scatter(xa, ya, s=20, c=co, marker='^')
ax.scatter(f,g, s=40, c=h,marker = 'o')
for i,txt in enumerate(yu):
    ax.annotate(txt, (f[i],g[i]))
for i, txt in enumerate(x):
    ax.annotate(txt, (xa[i], ya[i]))
    
fig.savefig('D:\\task3_iter2_a.jpg', dpi=fig.dpi)

figu,axu = plt.subplots()
axu.scatter(xa, ya, s=20, c=co, marker='^')
axu.scatter(fu,gu, s=40, c=h,marker = 'o')
for i,txt in enumerate(yuup):
    axu.annotate(txt, (fu[i],gu[i]))
for i, txt in enumerate(x):
    axu.annotate(txt, (xa[i], ya[i]))
figu.savefig('D:\\task3_iter2_b.jpg', dpi=figu.dpi)


