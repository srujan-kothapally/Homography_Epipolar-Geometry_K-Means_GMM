import numpy as np
from math import e,pi,sqrt
import cv2
import matplotlib.pyplot as plt


def kmeanscluster(k):
    img = cv2.imread('D:\\baboon.JPG')


    new_img = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])

  
    qua=[]
    poi=[]
    randomx =[]
    randomy =[]
    
    randomx = np.random.choice(512,k)
    randomy = np.random.choice(512,k)


    for i in range(k):
        poi.append((randomx[i],randomy[i]))

    for i in range(k):
        qua.append(img[poi[i][0]][poi[i][1]])
    qua = np.asarray(qua)

    loop=0

    def kmeans(x,uz,k,loop):
    
    
        sumx=0
        sumy=0
        sumz=0
        leng=0
    
 
        ux=np.array([[0.0 for j in range(3)] for i in range(k)])
        yes=0


        dist= []
        print(len(uz))
        for i in range(len(x)):
            gar=[]
            for j in range(len(uz)):
                distance = np.sqrt((uz[j][0]-x[i][0])**2 + (uz[j][1]-x[i][1])**2 + (uz[j][2]-x[i][2])**2)
            
                gar.append(distance)
            y= gar.index(min(gar))
        

            dist.append(y)


        for i in range(k):

        
            for j in range(len(dist)):
                
                if(dist[j]==i):                   
                
                    leng=leng+1
                    sumx=sumx+x[j][0]

                    sumy=sumy+x[j][1]
                    sumz=sumz+x[j][2]
                
            ux[i][0]=(sumx/leng)
            ux[i][1]=(sumy/leng)
            ux[i][2]=(sumz/leng)
            leng=0
            sumx=0
            sumy=0
            sumz=0

        for i in range(k):
            for j in range(3):
                if(int(ux[i][j])==int(uz[i][j])):
                    yes +=1


        if(yes == k*k or loop == 13):
            print("yes")
            return dist,ux
        else:
            loop=loop+1
#        return dist,ux
            print("rec")
            return kmeans(x,ux,k,loop)
       
        

    di,yu = kmeans(new_img,qua,k,loop)

    di = np.asarray(di)

    print(yu)

    for i in range(len(new_img)):
        new_img[i] = yu[di[i]]

    ts=np.reshape(new_img,(512,512,3))

    cv2.imwrite('D://task3_baboon_'+str(k)+'.jpg',np.asarray(ts,dtype=np.uint8))

    return

k=[3,5,10,20]
for i in k:
    print("for k= "+str(i))
    print ("PLEASE WAIT FOR SOME TIME")
    kmeanscluster(i)
print("DONE")


