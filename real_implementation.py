from GQA.quantum.compact_knapsack_qiskit import Compact_knapsack_qiskit

istance = 'GQA/knapsack_instances/low-dimensional/f3_l-d_kp_4_20'
generations = 5
delta = 0.025
num_pop = 1
verbose = 1
mutation_rate  = 0.05
seed = 1

alg = Compact_knapsack_qiskit(generations,delta,num_pop,istance,verbose,mutation_rate,seed)
alg.run()