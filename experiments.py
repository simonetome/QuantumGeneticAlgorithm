from GQA.functions.peaks import Peaks
from GQA.functions.rastrigin import Rastrigin
from GQA.functions.eggholder import Eggholder
from GQA.quantum.gqa_function import GQA_function
from GQA.utils.plotutils import plotList,average_evals
from tqdm import tqdm

# r = Rastrigin(-5.12,5.12,-5.12,5.12)
# r.plot_()

# e = Eggholder(-512,512,-512,512)
# e.plot_()

# p = Peaks(-3,3,-3,3)
# p.plot_()


l = []
params = []

runs = 50
generations = 200
num = 64
num_pop = 16
deltas = [0.0025,0.025,0.25,0.5]
seeds = range(runs)
verbose = 0
function = "rastrigin" #  | "eggholder" | "rastrigin" | "peaks"
decay = 0
mutation_rate = 0.01
crossover_probability = 0.5
avg_evals = {}
bst_evals = {}

for i in tqdm(deltas):
    evals = []
    avg_evals[i] = 0
    bst_evals[i] = 0

    tmp = []
    for s in tqdm(seeds):
        gqa = GQA_function(generations,num,num_pop,i,s,verbose,function,decay,mutation_rate,crossover_probability)
        gqa.run()
        evals.append(gqa.best_evaluations)
        tmp.append(gqa.best_fitness)

    l.append(average_evals(evals))
    avg_evals[i] = sum(tmp)/len(tmp)
    bst_evals[i] = min(tmp)
    params.append(i)


plotList(l,params,generations)
print(avg_evals)
print(bst_evals)



