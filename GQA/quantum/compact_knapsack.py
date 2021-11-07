from numpy import mat
from GQA.quantum.population import Population
from operator import itemgetter
from GQA.functions.knapsack import Knapsack
import math
import random 
from tqdm import tqdm
from matplotlib import pyplot as plt



class Compact_knapsack:
    def __init__(self,generations,delta,num_pop,istance,verbose,mutation_rate,seed,penalty):
        random.seed(seed)
        self.penalty = penalty

        self.generations = generations 
        self.generation = 0
        self.num_pop = num_pop
        self.verbose = verbose
        self.best_evaluation = -999999999
        self.mutation_rate = mutation_rate 
        self.best_evaluations = []
        self.knapsack = Knapsack(istance,penalty=self.penalty)
        self.num = self.knapsack.size
        self.population = Population(1,self.num) 

        self.measurements = {}
        for i in range(self.num_pop):
            self.measurements[i] = list()

        self.evaluations = {}
        self.delta = math.pi*delta
        self.best_chromosome = []
        self.rotations = [[]]
        for i in range(self.num):
            self.rotations[0].append(0)
    
    def run(self):
        while self.generation < self.generations:
            self.run_generation()
            self.generation += 1

            if(self.verbose):
                print(f"best fitness: {self.best_evaluation} ")
            
            self.best_evaluations.append(self.best_evaluation)
    
    def run_generation(self):
        # --------------- QUANTUM SUBROUTINE -----------------

        for i in range(self.num_pop):
            self.population.reset()                                 # reset all the population: all num_pop chromosomes are a sequence of |0> genes
            self.population.H_barrier()                             # all genes pass through an Hadamard gate to get into |+> state
            self.population.U_barrier(self.rotations)               # all genes are rotated with an Ry(theta) gate with a theta equal to the accumalated value in the classical computing routine.  
            self.measurements[i] = self.population.measure_compact()
        
        # -------------- CLASSICAL COMPUTATION ---------------

        self.evaluation()                                       # evaluate all the chromosomes
        self.update()
        self.mutation()

    def evaluation(self):

        #################################
        #           REPAIR              #
        #################################
        #self.repair()
        
    
        for i in self.measurements:
            self.evaluations[i] = self.knapsack.calculate_revenue(l = self.measurements[i])
        
        max_index = max(self.evaluations.items(), key=itemgetter(1))[0]
        max_value = self.evaluations[max_index]

        if(max_value > self.best_evaluation):      
            self.best_chromosome = self.measurements[max_index]
            self.best_evaluation = self.evaluations[max_index]
    
    def update(self):
        
        for j in range(self.num):     # j index of the gene
            sum = 0
            for i in self.measurements.keys():
                sum += self.measurements[i][j]
            sum /= self.num_pop

            b = self.best_chromosome[j]

            if(b == 0 and sum >= 0.5):
                self.rotations[0][j] = (self.rotations[0][j] - self.delta)
            elif (b == 1 and sum <= 0.5):
                self.rotations[0][j] = (self.rotations[0][j] + self.delta) 
    
    def mutation(self):
        for i in range(self.num):
            x = random.uniform(0,1)

            if(x < self.mutation_rate):
                self.rotations[0][i] *= -1
    
    def repair(self):
        for i in self.measurements: # i index of the chromosome
            repaired_chromosome, deleted_genes = self.knapsack.repairment(self.measurements[i])
            self.measurements[i] = repaired_chromosome[:]
            # reset rotations for deleted genes
            # if(len(deleted_genes) > 0):
            #     for j in deleted_genes:
            #         self.rotations[i][j] = 0
    

