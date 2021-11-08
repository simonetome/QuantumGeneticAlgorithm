

from GQA.quantum.gqa_function import GQA_function
from tqdm import tqdm

mutations = [0.0,0.01,0.025]
crossovers = [0.0,0.25,0.5,0.75]


runs = 30
generations = 200
num = 64
num_pop = 16 # setted to 8 for Table 7, to 16 for Table 8 

seeds = range(runs)
verbose = 0
function = "peaks"
decay = 0 
delta = 0.025

evals = {}

for c in tqdm(crossovers):
    for m in tqdm(mutations):
        sum = 0
        for s in seeds:
            gqa = GQA_function(generations,num,num_pop,delta,s,verbose,function,decay,m,c)
            gqa.run()
            sum += gqa.best_fitness
        
        evals[str(c)+str(m)] = sum/runs

print(evals)


seeds = range(runs)
verbose = 0
function = "eggholder"
decay = 0 
delta = 0.025

evals = {}

for c in tqdm(crossovers):
    for m in tqdm(mutations):
        sum = 0
        for s in seeds:
            gqa = GQA_function(generations,num,num_pop,delta,s,verbose,function,decay,m,c)
            gqa.run()
            sum += gqa.best_fitness
        
        evals[str(c)+str(m)] = sum/runs

print(evals)

seeds = range(runs)
verbose = 0
function = "rastrigin"
decay = 0 
delta = 0.025

evals = {}

for c in tqdm(crossovers):
    for m in tqdm(mutations):
        sum = 0
        for s in seeds:
            gqa = GQA_function(generations,num,num_pop,delta,s,verbose,function,decay,m,c)
            gqa.run()
            sum += gqa.best_fitness
        
        evals[str(c)+str(m)] = sum/runs

print(evals)

