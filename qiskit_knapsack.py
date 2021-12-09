from random import seed
from GQA.quantum.gqa_qiskit import GQA_qiskit
from GQA.quantum.gqa_knapsack import GQA_knapsack
from GQA.utils.plotutils import average_evals, plotBestEval, plotKnapsackComparison
from tqdm import tqdm 

# parameters

generations = 100
num_pop = 4
plots = []
knapsack_definition = "GQA/knapsack_instances/low-dimensional/f5_l-d_kp_15_375" 
instances = [
                "GQA/knapsack_instances/low-dimensional/f7_l-d_kp_7_50",
                "GQA/knapsack_instances/low-dimensional/f1_l-d_kp_10_269",
                "GQA/knapsack_instances/low-dimensional/f6_l-d_kp_10_60",
                "GQA/knapsack_instances/low-dimensional/f5_l-d_kp_15_375",
            ]
delta = 0.025
penalty = None
runs = 10
seeds = range(runs)
verbose = 0
crossover_probability = 0.5
mutation_rate = 0.025
shots = 1

best_evaluations = [0]*generations
best = 0
best_25 = 0
best_50 = 0 

best_ideal = 0
best_ideal_25 = 0
best_ideal_50 = 0
best_evaluations_ideal = [0]*generations

for instance in tqdm(instances):
    for s in tqdm(seeds):
        gqa = GQA_qiskit(generations,num_pop,instance,delta,penalty,s,verbose,crossover_probability,mutation_rate,shots)
        gqa.run()
        for i in range(generations):
            best_evaluations[i] += gqa.best_evaluations[i]
        
        if(gqa.best_evaluations[-1] > best):
            best = gqa.best_evaluations[-1]
        
        if(gqa.best_evaluations[24] > best_25):
            best_25 = gqa.best_evaluations[24]

        if(gqa.best_evaluations[49] > best_50):
            best_50 = gqa.best_evaluations[49]

    for i in range(generations):
        best_evaluations[i] /= runs 


    for s in tqdm(seeds):
        gqa = GQA_knapsack(generations,num_pop,instance,delta,penalty,s,verbose,crossover_probability,mutation_rate)
        gqa.run()

        for i in range(generations):
            best_evaluations_ideal[i] += gqa.best_evaluations[i]
        
        if(gqa.best_evaluations[-1] > best_ideal):
            best_ideal = gqa.best_evaluations[-1]
        
        if(gqa.best_evaluations[24] > best_ideal_25):
            best_ideal_25 = gqa.best_evaluations[24]

        if(gqa.best_evaluations[49] > best_ideal_50):
            best_ideal_50 = gqa.best_evaluations[49]

        print(gqa.best_evaluations)

    for i in range(generations):
        best_evaluations_ideal[i] /= runs 



    plotKnapsackComparison(best_evaluations,best_evaluations_ideal,instance)

    print(f"For instance {instance}")

    print("at 25 generations: ")
    print(f"Best evaluation qiskit found is {best_25}")
    print(f"Average is: {best_evaluations[24]}")
    print(f"Best evaluation ideal found is {best_ideal_25}")
    print(f"Average is: {best_evaluations_ideal[24]}")


    print("at 50 generations: ")
    print(f"Best evaluation qiskit found is {best_50}")
    print(f"Average is: {best_evaluations[49]}")
    print(f"Best evaluation ideal found is {best_ideal_50}")
    print(f"Average is: {best_evaluations_ideal[49]}")


    print("At 100 generations: ")
    print(f"Best evaluation qiskit found is {best}")
    print(f"Average is: {best_evaluations[-1]}")
    print(f"Best evaluation ideal found is {best_ideal}")
    print(f"Average is: {best_evaluations_ideal[-1]}")





