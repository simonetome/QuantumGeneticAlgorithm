# Script to make a quick comparison between GQA and a classical implementation.

from GQA.classical.ga_function import GA_function
from GQA.quantum.gqa_function import GQA_function
from GQA.utils.plotutils import plotComparison,average_evals,plotComparison
from tqdm import tqdm

l = []
params = []

runs = 30
generations = 200
num = 64
num_pop = 32
delta = 0.025
seeds = range(runs)
verbose = 0
function = "eggholder"
decay = 0 # no decay for delta theta
mutation_rate = 0.01
crossover_probability = 0.5


evals = []
for s in tqdm(seeds):
    ga = GA_function(generations,num,num_pop,mutation_rate,s,verbose,function,crossover_probability)
    ga.run()
    evals.append(ga.best_evaluations)

l.append(average_evals(evals))

evals = []
for s in tqdm(seeds):
    gqa = GQA_function(generations,num,num_pop,delta,s,verbose,function,decay,mutation_rate,crossover_probability)
    gqa.run()
    evals.append(gqa.best_evaluations)

l.append(average_evals(evals))

plotComparison(l,generations)



