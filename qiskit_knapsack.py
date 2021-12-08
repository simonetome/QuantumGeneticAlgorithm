from random import seed
from GQA.quantum.gqa_qiskit import GQA_qiskit
from GQA.quantum.gqa_knapsack import GQA_knapsack
from GQA.utils.plotutils import average_evals, plotBestEval, plotKnapsackComparison
from tqdm import tqdm 

# parameters

generations = 50
num_pop = 4
knapsack_definition = "GQA/knapsack_instances/low-dimensional/all1_15" 
delta = 0.025
penalty = None
runs = 10
seeds = range(runs)
verbose = 0
crossover_probability = 0.5
mutation_rate = 0.05
shots = 1

best_evaluations = [0]*generations
best = 0

best_ideal = 0
best_evaluations_ideal = [0]*generations


for s in tqdm(seeds):
    gqa = GQA_qiskit(generations,num_pop,knapsack_definition,delta,penalty,s,verbose,crossover_probability,mutation_rate,shots)
    gqa.run()
    for i in range(generations):
        best_evaluations[i] += gqa.best_evaluations[i]
    
    if(gqa.best_evaluations[-1] > best):
        best = gqa.best_evaluations[-1]

for i in range(generations):
    best_evaluations[i] /= runs 


for s in tqdm(seeds):
    gqa = GQA_knapsack(generations,num_pop,knapsack_definition,delta,penalty,s,verbose,crossover_probability,mutation_rate)
    gqa.run()

    for i in range(generations):
        best_evaluations_ideal[i] += gqa.best_evaluations[i]
    
    if(gqa.best_evaluations[-1] > best_ideal):
        best_ideal = gqa.best_evaluations[-1]
    print(gqa.best_evaluations)

for i in range(generations):
    best_evaluations_ideal[i] /= runs 



plotKnapsackComparison(best_evaluations,best_evaluations_ideal)

print(f"Best evaluation qiskit found is {best}")
print(f"Average is: {best_evaluations[-1]}")

print(f"Best evaluation ideal found is {best_ideal}")
print(f"Average is: {best_evaluations_ideal[-1]}")

# f7_l-d_kp_7_50 optima is 107 [updated with fix]
# Best evaluation qiskit found is 107.0
# Average is: 107.0
# Best evaluation ideal found is 107.0
# Average is: 107.0

# optima is 295 for f1_l-d_kp_10_269[updated with fix] 
# Best evaluation qiskit found is 295.0
# Average is: 293.9
# Best evaluation ideal found is 295.0
# Average is: 294.4

# 100 gen:[updated with fix] 
# Best evaluation qiskit found is 295.0
# Average is: 294.8
# Best evaluation ideal found is 295.0
# Average is: 294.8



# f6_l-d_kp_10_60 optima is 52[updated with fix] 
# Best evaluation qiskit found is 52.0
# Average is: 51.9
# Best evaluation ideal found is 52.0
# Average is: 52.0

#100 gen fixed[updated with fix] 
# Best evaluation qiskit found is 52.0
# Average is: 52.0
# Best evaluation ideal found is 52.0
# Average is: 52.0



#100

# f5_l-d_kp_15_375 opt is 481,0694 [updated with fix] 
# Best evaluation qiskit found is 481.069368
# Average is: 465.25650750000005
# Best evaluation ideal found is 481.069368
# Average is: 476.93777230000006

# 100gen:[updated with fix] 
# Best evaluation qiskit found is 481.069368
# Average is: 472.70149790000005
# Best evaluation ideal found is 481.069368
# Average is: 478.6877036000001



# all 1 15 qubits [updated with fix] 
# Best evaluation qiskit found is 14.0
# Average is: 12.4
# Best evaluation ideal found is 14.0
# Average is: 13.2 

# 100 [updated with fix] 
# Best evaluation qiskit found is 14.0
# Average is: 12.7
# Best evaluation ideal found is 14.0
# Average is: 13.6