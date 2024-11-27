from qiskit_aer import AerSimulator


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
from collections import OrderedDict


from quark import Task


# 实例化任务管理器
token = 'xWfrrcc`iCURRLkXKb7FrO398rmFKIhl1sKDryIqYsf/14NyRkO2BkM2hEP1hENzN{OypkJxiY[jxjJ2JkPyRkP1FEJyJUMxFUM1JENzJjPjRYZqKDMj53ZvNXZvNYbyGnZBKXfwW4[jpkJzW3d2Kzf'
tmgr = Task(token)


def fill(data):
    n = len(list(data.keys())[0])
    full_results = {}
    for i in range(2**n):
        full_results[bin(i)[2:].zfill(n)] = 0
    for key in data.keys():
        full_results[key] = data[key]
    return full_results



def compare_plot(data_dict, title:Optional[str] = None, width: Optional[float] = 0.3):

    colors = ['C2', 'C1', 'skyblue']
    # labels = ['theoretical', 'experiment', 'byes']
    data_all = {}
    probs_all = {}
    
    for name , data in data_dict.items():
        data_all[name] = fill(data)
        # data_all[name] = {k: data_all[name][k] for k in sorted(data_all[name])}
        probs_all[name] = data_all[name].values()
    x = np.arange(len(data))

    bitstrs = list(data.keys())
    bitstrs = [str(i) for i in bitstrs]

    plt.figure(figsize=(6, 3))
    for i, name in enumerate(probs_all):
        plt.bar(x+width*i, probs_all[name], width=width,color=colors[i],label = name)

    plt.legend()
    plt.xticks(x+width, labels=bitstrs)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.title(title)
    plt.show()
    


def plot_probabilities(sampling_results, title:Optional[str] = None,
                       tab:Optional[str] = None):
    """
    Plot the probabilities from execute results.
    """
    # sampling_results = {k: sampling_results[k] for k in sorted(sampling_results)}
    # sampling_results = {k: sampling_results[k] for k in sorted(sampling_results, key=lambda x: int(x, 2))}
    n = len(list(sampling_results.keys())[0])
    full_results = {}
    for i in range(2**n):
        full_results[bin(i)[2:].zfill(n)] = 0
    for key in sampling_results.keys():
        full_results[key] = sampling_results[key]
    bitstrs = list(full_results.keys())
    bitstrs = [str(i) for i in bitstrs]
    probs = list(full_results.values())
    plt.figure(figsize=(6,3))
    plt.bar(range(len(probs)), probs, tick_label = bitstrs)
    plt.xticks(rotation=70)
    plt.ylabel("probabilities")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.title(title)
    plt.show()
 
def simulation_qis(qc,plot:bool):
    aersim = AerSimulator()
    result_ideal = aersim.run(qc).result()
    counts = result_ideal.get_counts()
    n = len(qc.qubits)
    full_result = {}
    for i in range(2**n):
        full_result[bin(i)[2:].zfill(n)] = 0
    for i in counts.keys():
        full_result[i] = counts[i]
    # full_result =  { k[::-1] : v for k,v in full_result.items()}
    if plot == True:
        plot_probabilities(full_result)
    return full_result

def simulation_quf(circ,plot:bool):
    from quafu import simulate
    counts = simulate(circ, shots=10240).counts
    # simulate(circ).plot_probabilities()
    # aersim = AerSimulator()
    # result_ideal = aersim.run(qc).result()
    # counts = result_ideal.get_counts()
    n = len(circ.used_qubits)
    full_result = {}
    for i in range(2**n):
        full_result[bin(i)[2:].zfill(n)] = 0
    for i in counts.keys():
        full_result[i] = counts[i]
    if plot == True:
        plot_probabilities(full_result)
    return full_result


def get_results(results_filename):
    df = pd.read_csv(results_filename,header=None)
    ids = df.iloc[:, 0].tolist()
    nc = len(ids)
    error = []
    results = {}
    for i in range(nc):
        try:
            data = tmgr.result(ids[i])['count']
            data_prime =  { k[::-1] : v for k,v in data.items()}
            results[i] = data_prime
            # results[i] = data
        except:
            error.append(i)
            print(i, ids[i])
    results = {index: value for index, value in enumerate(results.values())}

    return results, error

def fidelity(circ, data):
    a = simulation_quf(circ, plot = True)
    plot_probabilities(data, title='after byes')
    data_prob = {}
    s = np.sum(list(data.values()))
    for key, value in data.items():
        data_prob[key] = value/s

    data_idle = {}

    for key, value in a.items():
        data_idle[key] = value/10240
    fidelity = 0
    for key in data_idle.keys():
        try:
            fidelity += np.sqrt(data_idle[key]*data_prob[key])
        except:
            pass
    print(fidelity)
############################# Bayes read out
def map_qlisp(C, qmap) :
    Cp = [];
    for ins in C :
        ins1_mapped=None;
        if(type(ins[1]) == int) : ins1_mapped = qmap[ins[1]];
        elif(type(ins[1]) == tuple and len(ins[1]) == 2) : ins1_mapped = ( qmap[ins[1][0]],qmap[ins[1][1]]);
        elif(type(ins[1]) == tuple and len(ins[1]) > 2) : ins1_mapped = ( qmap[ins[1][0]],qmap[ins[1][1]],  qmap[ins[1][2]],qmap[ins[1][3]]);
        Cp.append( (ins[0] ,ins1_mapped ) )
    return Cp;


def find_pos(binary_str):
    pos_0 = []
    pos_1 = []
    for i, bit in enumerate(binary_str):
        if bit == '0':
            pos_0.append(i)
        else:
            pos_1.append(i)
    return pos_0, pos_1

def read_correct_cir(qmap, n):
    def add_x(qlisp_ins, q):
        qlisp_ins.append(('X', q))

    def add_i(qlisp_ins, q):
        qlisp_ins.append(('I', q))

    measures =  [ (("Measure",i) , i ) for i in range(n)] ; 

    read_circuits  = {i:[] for i in range(2**n)}

    for i in range(2**n):
        i_2 = bin(i)[2:].zfill(n)
        i_pos, x_pos = find_pos(i_2)
        for j in  range(n):
            if j in i_pos:
                add_i(read_circuits[i], j)
            if j in x_pos:
                add_x(read_circuits[i], j)
        read_circuits[i]=map_qlisp(read_circuits[i] , qmap )
        read_circuits[i] = read_circuits[i] + [ ("Barrier",  tuple([w for w in qmap.values()]))] + map_qlisp( measures ,qmap )
    return read_circuits


def get_read_mat(read_results,n):
    mat = np.zeros((2**n, 2**n), dtype=int)
    for i in range(2**n):
        for j in range(2**n):
            try:
                mat[i][j] = read_results[i][bin(j)[2:].zfill(n)[::-1]]
            except:
                mat[i][j] = 0
            # print(read_results[i][bin(j)[2:].zfill(4)])
    mat = mat/20480
    # mat = mat/10240
    read_mat = np.linalg.inv(mat) 
    return np.transpose(read_mat)

def read_correct(results:dict, read_mat, n): #多次实验
    l = len(results)
    results_mat = np.zeros((2**n, l))
    for i in range(l):
        for j in list(results[0].keys()):
            try:
                results_mat[int(j,2),i] = int(results[i][j])
            except:pass
    after_mat = np.dot(read_mat,results_mat)

    def re(i):
        res = {}
        for j in range(2**n):
            res[bin(j)[2:].zfill(n)] = after_mat[j][i]
        return res
    
    results_after = {}
    for c in range(l):
        results_after[str(c)] = re(c)
    return results_after


def r_correct(result:dict,read_mat, n): #单个count
    results_mat = np.zeros(2**n)
    for j in list(result.keys()):
        results_mat[int(j,2)] = int(result[j])
    after_mat = np.dot(read_mat,results_mat)
    
    res = {}
    for j in range(2**n):
        res[bin(j)[2:].zfill(n)] = after_mat[j]

    return res





