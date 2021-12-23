import numpy as np
p=np.array([[0,0.],[0,1],[1,1],[1,0]])
q=np.array([[0.3,0.3],[0,1],[1,1],[1,0]])
import matplotlib.image as py
img=py.imread('C:/Users/Administrator/Pictures/0.png')
u,v=img.shape[:2]
def f(i,j):
    return i+0.1*np.sin(2*np.pi*j)
def g(i,j):
    return j+0.1*np.sin(3*np.pi*i)
M=[]
N=[]
for i in range(u):
    for j in range(v):
        i0=i/u
        j0=j/v
        u0=int(f(i0,j0)*512)
        v0=int(g(i0,j0)*512)
        M.append(u0)
        N.append(v0)
m1,m2=max(M),max(N)
n1,n2=min(M),min(N)
r=np.zeros((m1-n1,m2-n2,4))
for i in range(u):
    for j in range(v):
        i0=i/u
        j0=j/v
        u0=int(f(i0,j0)*512)-n1-1
        v0=int(g(i0,j0)*512)-n2-1
        r[u0,v0]=img[i,j]
py.imsave('C:/Users/Administrator/Pictures/1.png',r)