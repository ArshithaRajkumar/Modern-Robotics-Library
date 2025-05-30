import numpy as np
import math

def nearzero(x):
    return x<1e-6

def normalize(x):
    b=math.sqrt(np.dot(x, x))
    d=x/b
    c=x/np.linalg.norm(x)
    return c

def rotinv(R):
    #RT = np.zeros((len(R), len(R)))
    RT=[[0 for _ in range(len(R))] for _ in range(len(R))]
    for a in range (len(R)):
        for b in range(len(R)):
            RT[a][b]=R[b][a]
    RT2 = np.transpose(R)                        
    return RT2

def vectoso3(w):
    w1, w2, w3 = w[0], w[1], w[2]
    return np.array([[0, -w3, w2],
                     [w3, 0, -w1],
                     [-w2, w1, 0]]) 

def so3tovec(W):
    return np.array([-W[1][2],W[0][2],-W[0][1]])

def axisang3(omghattheta):
    omega=normalize(omghattheta)
    theta=np.linalg.norm(omghattheta)
    return (omega,theta)

def matrixexp3(so3mat):
    so3vec=so3tovec(so3mat)
    omega=axisang3(so3vec)[0]
    theta=axisang3(so3vec)[1]
    R=np.identity(3)+np.sin(theta)*so3mat+(1-np.cos(theta))*np.dot(so3mat)
    return R

def matrixlog3(R):
    ctheta=(np.trace(R)-1)/2
    if ctheta>=1:
        return np.zeros(3)
    elif ctheta<=-1:
        if not nearzero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not nearzero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return vectoso3(np.pi * omg)
    else:
        theta=np.arccos(ctheta)
        omg=(R-R.T)/2*np.sin(theta)
        return (theta*omg)