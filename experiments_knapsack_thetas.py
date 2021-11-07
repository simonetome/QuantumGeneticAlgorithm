from GQA.quantum.gqa_knapsack import GQA_knapsack
from GQA.classical.ga_knapsack import GA_knapsack
from GQA.utils.plotutils import plotList,average_evals
from tqdm import tqdm
import os 

#self,generations,num,num_pop,delta,seed,verbose,function,decay,mutation_rate,crossover_probability

l = []
params = []

runs = 10
generations = 100
num_pop = 16
deltas = [0.025]
seeds = range(runs)
verbose = 0
istance = os.path.join('GQA','knapsack_instances','large_scale','knapPI_1_100_1000_1')
decay = 0
mutation_rate = 0.01
crossover_probability = 0.5

for i in tqdm(deltas):
    evals = []
    for s in tqdm(seeds):
        gqa = GQA_knapsack(generations,num_pop,istance,i,'quadratic',s,verbose,crossover_probability,mutation_rate)
        gqa.run()
        evals.append(gqa.best_evaluations)

    l.append(average_evals(evals))
    params.append(i)


plotList(l,params,generations)


#gqa.plot_history()


