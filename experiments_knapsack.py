from GQA.quantum.gqa_knapsack import GQA_knapsack
from GQA.classical.ga_knapsack import GA_knapsack
import os
from tqdm import tqdm

crossover_probability = 0.5                                # rate of crossver
generations = 200                                           # number of generations for the algorithm 
runs = 10
seeds = range(runs)
n = 32                                                      # number of chromosomes 
mutation_rate = 0.01                                       # each gene mutates with a chance of 2.5% for each generation (avoid to get stuck in local optimals)
delta_theta = 0.025                                         # delta_theta parameter for the quantum rotation gates (the increment) -> 0.05pi
s = 0                                                       # seed: used for random functions
verbose = 0                                                 # 0: don't print anything, 1: print results for each generation
instances = os.listdir(os.path.join('GQA','knapsack_instances','large_scale'))         # file name (without extension)
decay = False       

results = {}

# Penalty none as we use repairment of the chromosome which is a more sophisticated optimization technique
for i in tqdm(instances):
    instance = os.path.join('GQA','knapsack_instances','large_scale',i)
    sum = 0
    gen = 0
    for s in tqdm(seeds):
        gqa = GQA_knapsack(generations = generations,num_pop=n,knapsack_definition=instance,delta=delta_theta,penalty= None,seed=s,verbose=0,crossover_probability=crossover_probability,mutation_rate=mutation_rate)
        gqa.run()
        sum += gqa.best_fitness
        gen += gqa.best_generation
    sum /= runs 
    gen /= runs
    results[i] = {}
    results[i]['res'] = sum
    results[i]['gen'] = gen


for i in tqdm(instances):
    instance = os.path.join('GQA','knapsack_instances','large_scale',i)
    sum = 0
    gen = 0
    for s in tqdm(seeds):
        ga = GA_knapsack(generations = generations,num_pop=n,knapsack_definition=instance,penalty= None,seed=s,verbose=0,crossover_probability=crossover_probability,mutation_rate=mutation_rate)
        ga.run()
        sum += ga.best_fitness
        gen += ga.best_generation
    sum /= runs 
    gen /= runs
    results[i] = {}
    results[i]['res'] = sum
    results[i]['gen'] = gen

print(results)