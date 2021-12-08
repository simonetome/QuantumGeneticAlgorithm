from GQA.quantum.compact_knapsack_qiskit import Compact_knapsack_qiskit
from GQA.utils.plotutils import plotBestEval
from tqdm import tqdm

istance = 'GQA/knapsack_instances/low-dimensional/f9_l-d_kp_5_80' # real optima is 130 
generations = 15
delta = 0.05
num_pop = 4
verbose = 1
mutation_rate  = 0.025
seed = 1
runs = 5
average_evaluations = [0]*generations
print_final = 0 

for i in tqdm(range(runs)):
    if (i == runs-1):
        print_final = 1
    alg = Compact_knapsack_qiskit(generations,delta,num_pop,istance,verbose,mutation_rate,seed,print_final)
    alg.run()

    for i in range(generations):
        average_evaluations[i] += alg.best_evaluations[i]

for i in range(generations):
    average_evaluations[i] /= runs 

plotBestEval(average_evaluations)
