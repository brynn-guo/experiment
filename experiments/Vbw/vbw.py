import  numpy as np ;

import random;
import pickle ;
import time ;
import matplotlib.pyplot as plt
import pandas as pd
import random
import json
from importlib import reload; 
from typing import Optional
from collections import OrderedDict
from quark import Task
import sys
import quark.circuit 

sys.path.append('/Users/brynn/work/gyb')
from gates import add_barrier,add_CZ,add_cnot,add_H,add_R,add_U3, mat_r
# reload(gates)


# 实例化任务管理器
token = 'xWfrrcc`iCURRLkXKb7FrO398rmFKIhl1sKDryIqYsf/14NyRkO2BkM2hEP1hENzN{OypkJxiY[jxjJ2JkPyRkP1FEJyJUMxFUM1JENzJjPjRYZqKDMj53ZvNXZvNYbyGnZBKXfwW4[jpkJzW3d2Kzf'
tmgr = Task(token)

def U_2(qlisp_ins, circ, q0, q1, phi):
    add_cnot(qlisp_ins, circ, q0, q1)
    add_R('Rz', qlisp_ins, circ, phi, q1)
    add_cnot(qlisp_ins, circ, q0, q1)
    

def U_3(qlisp_ins, circ, q0, q1, q2, phi):
    add_cnot(qlisp_ins, circ, q0, q1)
    # add_barrier(qlisp_ins, circ, (0,1,2,3))
    add_cnot(qlisp_ins, circ, q1, q2)
    add_R('Rz', qlisp_ins, circ, phi, q2)
    add_cnot(qlisp_ins, circ, q1, q2)
    add_cnot(qlisp_ins, circ, q0, q1)


def vbw_single(qlisp_ins, circ, j, g, h, lam):
    
    U_2(qlisp_ins, circ, j-1 ,j, 2)
    add_barrier(qlisp_ins, circ, (0,1,2,3))
    add_R('Rx', qlisp_ins, circ, -2*g, j-1)
    add_R('Rz', qlisp_ins, circ, -2*h, j-1)
    add_H(qlisp_ins, circ, j-1)
    add_barrier(qlisp_ins, circ, (0,1,2,3))
    U_3(qlisp_ins, circ, j-1, j, j+1, -lam*2)
    # # add_barrier(qlisp_ins, circ, (0,1,2,3))
    add_H(qlisp_ins, circ, j-1)
    add_H(qlisp_ins, circ, j+1)
    # add_barrier(qlisp_ins, circ, (0,1,2,3))
    U_3(qlisp_ins, circ, j-1, j, j+1, -lam*2)
    # add_barrier(qlisp_ins, circ, (0,1,2,3))
    add_H(qlisp_ins, circ, j+1)

def U_2_simple(qlisp_ins, circ, q0, q1, phi):
    mat_h = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
    mat_rz = mat_r('Rz', phi)
    flag = quark.circuit.u3_decompose(np.dot(np.dot(mat_h,mat_rz), mat_h))
    
    add_H(qlisp_ins, circ, q1)
    # qlisp_ins.append((('Delay', 100e-9), q0))
    add_CZ(qlisp_ins, circ, q0, q1)
    add_U3(qlisp_ins, circ, q1, flag[0], flag[1], flag[2])
    add_CZ(qlisp_ins, circ, q0, q1)
    add_H(qlisp_ins, circ, q1)

def U_3_simple(qlisp_ins, circ, q0, q1, q2, phi):

    mat_h = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
    mat_rz = mat_r('Rz', phi)
    flag = quark.circuit.u3_decompose(np.dot(np.dot(mat_h,mat_rz), mat_h))
    # qlisp_ins.append((('Delay', 200),q2))
    add_H(qlisp_ins, circ, q1)
    add_CZ(qlisp_ins, circ, q0, q1)
    add_barrier(qlisp_ins, circ, (q0,q1,q2, 3))
    add_H(qlisp_ins, circ, q1)

    # qlisp_ins.append((('Delay', 480),q2))

    add_H(qlisp_ins, circ, q2)
    

    #######====================#######
    # add_barrier(qlisp_ins, circ, (q0,q1,q2,3))
    # qlisp_ins.append((('Delay', 250), q0))
    # add_R('Rx', qlisp_ins, circ, np.pi, q0)
    # qlisp_ins.append((('Delay', 250), q0))
    # add_R('Rx', qlisp_ins, circ, -np.pi, q0)
    #######====================#######

    add_CZ(qlisp_ins, circ, q1, q2)
    add_U3(qlisp_ins, circ, q2, flag[0], flag[1], flag[2])
    add_CZ(qlisp_ins, circ, q1, q2)

    ######====================#######
    # qlisp_ins.append((('Delay', 100), q0))
    # add_R('Rx', qlisp_ins, circ, np.pi, q0)
    # add_R('Rx', qlisp_ins, circ, -np.pi, q0)
    ######====================#######

    # add_H(qlisp_ins, circ, q1)  ##开始变形
    # add_H(qlisp_ins, circ, q2)
    # add_CZ(qlisp_ins, circ, q0, q1)
    # add_H(qlisp_ins, circ, q1)

def vbw_single_simple(qlisp_ins, circ, j, g, h, lam):
    # mat_rx = mat_r('Rx', -2*g)
    # mat_rz = mat_r('Rz', -2*h)
    # mat_h = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
    # flag = quark.circuit.u3_decompose(np.dot(np.dot(mat_rx,mat_rz), mat_h))
    
    U_2_simple(qlisp_ins, circ, j-1 ,j, 2)
    add_barrier(qlisp_ins, circ, (0,1,2,3))
    add_R('Rx', qlisp_ins, circ, -2*g, j-1)
    add_R('Rz', qlisp_ins, circ, -2*h, j-1)
    add_H(qlisp_ins, circ, j-1)

    add_barrier(qlisp_ins, circ, (0,1,2,3))
    U_3_simple(qlisp_ins, circ, j-1, j, j+1, -lam*2)

    add_H(qlisp_ins, circ, j-1)
    add_H(qlisp_ins, circ, j+1)
    add_barrier(qlisp_ins, circ, (0,1,2,3))
    U_3_simple(qlisp_ins, circ, j-1, j, j+1, -lam*2)
    # add_barrier(qlisp_ins, circ, (0,1,2,3))
    add_H(qlisp_ins, circ, j+1)

def vbw_single_CPMG(qlisp_ins, circ, j, g, h, lam):

    U_2_simple(qlisp_ins, circ, j-1 ,j, 2)

    #######====================#######     # u2部分dd
    # qlisp_ins.append((('Delay', 100),j+1))
    # add_R('Rx', qlisp_ins, circ, np.pi, j+1)
    # add_R('Rx', qlisp_ins, circ, -np.pi, j+1)
    #######====================#######

    add_barrier(qlisp_ins, circ, (0,1,2,3))
    add_R('Rx', qlisp_ins, circ, -2*g, j-1)
    add_R('Rz', qlisp_ins, circ, -2*h, j-1)
    add_H(qlisp_ins, circ, j-1)

    # add_barrier(qlisp_ins, circ, (0,1,2,3))
    U_3_simple(qlisp_ins, circ, j-1, j, j+1, -lam*2)
  

    #######====================#######
    # qlisp_ins.append((('Delay', 200),j))
    #######====================#######   
    # add_barrier(qlisp_ins, circ, (0,1,2,3))
    # add_H(qlisp_ins, circ, j-1)
    # add_H(qlisp_ins, circ, j) # test 
    # add_H(qlisp_ins, circ, j+1)
    # # add_barrier(qlisp_ins, circ, (0,1,2,3))
    # U_3_simple(qlisp_ins, circ, j-1, j, j+1, -lam*2)

    # add_H(qlisp_ins, circ, j+1)
