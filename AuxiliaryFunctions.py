import numpy as np
import random
from math import log2

np.random.seed(19680801)

def randdistr(nodes):
    p = np.zeros(pow(2,nodes))
    for i in range(pow(2,nodes)):
        p[i] = 1 #random.randint(0,10000)
    p = p/np.sum(p)
    return p

def rand_cond_distr(nodes, cond_nodes):
    p = np.zeros((nodes, 2*pow(2,cond_nodes)))
    for j in range(nodes):
        for i in range(2*pow(2,cond_nodes)):
            p[j][i] =  random.randint(0,10000) # 100 + random.randint(0,100)
        p[j] = p[j]/np.sum(p[j])
    p_cond = np.copy(p)
    for j in range(nodes):
        pcond = np.zeros(pow(2,cond_nodes))
        for i in range(2*pow(2,cond_nodes)):
            pcond[i//2] = pcond[i//2] + p[j][i]
        for i in range(2 * pow(2, cond_nodes)):
            p_cond[j][i] = p[j][i] / pcond[i//2]
    return p_cond

def rand_cond_distr2(nodes, cond_nodes):
    p = np.zeros((nodes, 2*pow(2,cond_nodes)))
    for j in range(nodes):
        for i in range(2*pow(2,cond_nodes)):
            p[j][i] = 1 #random.randint(0,10000)
        p[j] = p[j]/np.sum(p[j])
    p_cond = np.copy(p)
    for j in range(nodes):
        pcond = np.zeros(pow(2,cond_nodes))
        for i in range(2*pow(2,cond_nodes)):
            pcond[i//2] = pcond[i//2] + p[j][i]
        for i in range(2 * pow(2, cond_nodes)):
            p_cond[j][i] = p[j][i] / pcond[i//2]
    return p_cond

def kl(y,x):
    sumkl = 0.0
    for i in range(len(y)):
        if y[i] > 0:
            if x[i] > 0:
                sumkl = sumkl + y[i] * (log2(y[i]) - log2(x[i]))
            else:
                sumkl = sumkl  +  100
    return sumkl

def getIndex(last_s, last_c, last_a):
    list = np.append( np.array(last_s).astype(int),np.array( last_c).astype(int))
    list = np.append(list, np.array(last_a).astype(int))
    string = ''.join([str(item) for item in list])
    return int(string,2)



