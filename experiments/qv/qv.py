





def get_ideals(circuits_filename, error):
    with open(circuits_filename, 'r') as file:
        circuits = json.load(file)
    circuits_qiskit = {}
    ideals = {}
    for i in range(len(circuits)):
        circuits_qiskit[i] = qiskitQC.from_qasm_str(circuits[str(i)]['qasm'])
        # circuits_qiskit[i].measure_all()
        circuits_qiskit[i].barrier([0,1,2,3])
        circuits_qiskit[i].measure([0,1,2,3],[0,1,2,3])
        ideals[i] = simulation_4(circuits_qiskit[i], plot = False)
        # ideals[i] =  { k[::-1] : v for k,v in ideals[i].items()} ###一会注释掉
    error_set = set(error)
    ideals_new = {key: value for key, value in ideals.items() if key not in error_set}
    ideals_new = {index: value for index, value in enumerate(ideals_new.values())}
    return ideals_new




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



def get_heavy_outputs(result,ideal):

    def heavy_outputs(ideal: OrderedDict):
        sorted_ideal = OrderedDict(sorted(ideal.items(), key=lambda x: x[1]))
        for _ in range(len(ideal)//2):
            key,value = sorted_ideal.popitem(last=False)
        # nheavies = np.sum(list(sorted_ideal.values()))
        return list(sorted_ideal.keys())

    heavy_ideal = heavy_outputs(ideal)

    def check_threshold(heavy_ideal:list, result:OrderedDict, nshots:int):
        nheavies = 0
        for i in result.keys(): 
            if i in heavy_ideal:
                nheavies += result[i]
        nshots = np.sum(list(result.values()))
        return nheavies/nshots

    return check_threshold(heavy_ideal=heavy_ideal, result=result,nshots=2000)