# script used to make hyperparameter optimization

from GQA.quantum.gqa_function import GQA_function
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from numpy import arange
import os

#------ plots ------------

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


#----- retrieve kanpsack ----

instance = os.path.join(os.getcwd(),"Code","GQA","knapsack_instances","large_scale","knapPI_1_100_1000_1")

#----- delta theta --------

generations = 200
mutation_rate = 0.01
delta_theta_to_plot = [0.005,0.075,0.5]
delta_theta = [0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.4,0.5]
trials = 30
seeds = range(0,trials)
crossover_probability = 0.75
num_pop = 40
num = 64
evaluations = {}
# performing benchmark

for d in tqdm(delta_theta):
    
    avg_evaluations = [0 for i in range(generations)]
    best = 0
    for s in tqdm(seeds):
        gqa = GQA_function(generations,num,num_pop,d,s,0,"eggholder",False,mutation_rate,crossover_probability)
        gqa.run()

        if(gqa.best_fitness > best):
            best = gqa.best_fitness
        
        for i in range(generations):
            avg_evaluations[i] += gqa.best_evaluations[i]
    
    for i in range(generations):
        avg_evaluations[i] /= trials

    if d in delta_theta_to_plot:
        ax1.plot(range(generations),avg_evaluations,label=f"Delta theta: {round(d,4)}pi")

        ax1.set_title(f'Delta theta')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('value')
        ax1.legend()

    evaluations[d] = {}
    evaluations[d]["best"] = best
    evaluations[d]["average"] = avg_evaluations[-1]


best = []
avg = []
deltas = []

for d in evaluations:
    deltas.append(d)
    avg.append(evaluations[d]["average"])
    best.append(evaluations[d]["best"])


ax2.plot(deltas,avg,label=f"Average results")
ax2.plot(deltas,best,label=f"Best results")
ax2.set_title(f'Delta theta')
ax2.set_xlabel('Deltas')
ax2.set_ylabel('Value')
ax2.legend()


plt.show()

#--------------------- mutation rate ------------------------------

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

generations = 200
mutation_rate = [0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
mutation_rate_to_plot = [0.005,0.01,0.05,0.2]
delta_theta = 0.15
trials = 30
num = 64
seeds = range(0,trials)
crossover_probability = 0.75
num_pop = 40
evaluations = {}
# performing benchmark

for m in tqdm(mutation_rate):
    
    avg_evaluations = [0 for i in range(generations)]
    best = 0
    for s in tqdm(seeds):
        gqa = GQA_function(generations,num,num_pop,delta_theta,s,0,"eggholder",False,m,crossover_probability)
        gqa.run()

        if(gqa.best_fitness > best):
            best = gqa.best_fitness
        
        for i in range(generations):
            avg_evaluations[i] += gqa.best_evaluations[i]
    
    for i in range(generations):
        avg_evaluations[i] /= trials

    if m in mutation_rate_to_plot:
        ax1.plot(range(generations),avg_evaluations,label=f"Mutation rate: {round(m,4)}")

        ax1.set_title(f'Mutation rate')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('value')
        ax1.legend()

    evaluations[m] = {}
    evaluations[m]["best"] = best
    evaluations[m]["average"] = avg_evaluations[-1]


best = []
avg = []
mutations = []

for m in mutation_rate:
    mutations.append(m)
    avg.append(evaluations[m]["average"])
    best.append(evaluations[m]["best"])


ax2.plot(mutations,avg,label=f"Average results")
ax2.plot(mutations,best,label=f"Best results")
ax2.set_title(f'Mutation rate')
ax2.set_xlabel('Mutation rate')
ax2.set_ylabel('Value')
ax2.legend()


plt.show()


#---------------------- population number -----------------------------

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

generations = 200
mutation_rate = 0.2
delta_theta = 0.15
trials = 30
num = 64
seeds = range(0,trials)
crossover_probability = 0.75
num_pop = [8,16,20,32,40,64,80,100]
num_pop_to_plot = [8,20,40,100]
evaluations = {}
# performing benchmark

for n in tqdm(num_pop):
    
    avg_evaluations = [0 for i in range(generations)]
    best = 0
    for s in tqdm(seeds):
        gqa = GQA_function(generations,num,n,delta_theta,s,0,"eggholder",False,mutation_rate,crossover_probability)
        gqa.run()

        if(gqa.best_fitness > best):
            best = gqa.best_fitness
        
        for i in range(generations):
            avg_evaluations[i] += gqa.best_evaluations[i]
    
    for i in range(generations):
        avg_evaluations[i] /= trials

    if n in num_pop_to_plot:
        ax1.plot(range(generations),avg_evaluations,label=f"Population size: {round(n,4)}")

        ax1.set_xlabel('Generation')
        ax1.set_ylabel('value')
        ax1.legend()

    evaluations[n] = {}
    evaluations[n]["best"] = best
    evaluations[n]["average"] = avg_evaluations[-1]


best = []
avg = []
mutations = []

for n in num_pop:
    mutations.append(n)
    avg.append(evaluations[n]["average"])
    best.append(evaluations[n]["best"])


ax2.plot(mutations,avg,label=f"Average results")
ax2.plot(mutations,best,label=f"Best results")
ax2.set_xlabel('Population size')
ax2.set_ylabel('Value')
ax2.legend()


plt.show()

#------------ crossover rate ---------------

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

generations = 200
mutation_rate = 0.2
delta_theta = 0.15
trials = 30
num = 64
seeds = range(0,trials)
crossover_probability = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
crossover_probability_to_plot = [0.1,0.2,0.4,0.6,0.9]
num_pop = 40
evaluations = {}
# performing benchmark

for c in tqdm(crossover_probability):
    
    avg_evaluations = [0 for i in range(generations)]
    best = 0
    for s in tqdm(seeds):
        gqa = GQA_function(generations,num,num_pop,delta_theta,s,0,"eggholder",False,mutation_rate,c)
        gqa.run()

        if(gqa.best_fitness > best):
            best = gqa.best_fitness
        
        for i in range(generations):
            avg_evaluations[i] += gqa.best_evaluations[i]
    
    for i in range(generations):
        avg_evaluations[i] /= trials

    if c in crossover_probability_to_plot:
        ax1.plot(range(generations),avg_evaluations,label=f"Population size: {round(c,4)}")

        ax1.set_xlabel('Generation')
        ax1.set_ylabel('value')
        ax1.legend()

    evaluations[c] = {}
    evaluations[c]["best"] = best
    evaluations[c]["average"] = avg_evaluations[-1]


best = []
avg = []
crossovers = []

for c in crossover_probability:
    crossovers.append(c)
    avg.append(evaluations[c]["average"])
    best.append(evaluations[c]["best"])


ax2.plot(crossovers,avg,label=f"Average results")
ax2.plot(crossovers,best,label=f"Best results")
ax2.set_xlabel('Crossover probability')
ax2.set_ylabel('Value')
ax2.legend()


plt.show()