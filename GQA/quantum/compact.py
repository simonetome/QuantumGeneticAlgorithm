from numpy import mat
from GQA.quantum.population import Population
from operator import itemgetter
from GQA.functions.peaks import Peaks
from GQA.functions.eggholder import Eggholder
from GQA.functions.multimodal import Multimodal
from GQA.functions.crossintray import CrossInTray
from GQA.functions.rastrigin import Rastrigin
from GQA.functions.function import Function
import math
import random 
from tqdm import tqdm
from matplotlib import pyplot as plt



class Compact:
    def __init__(self,generations,delta,num,num_pop,function,verbose,mutation_rate,seed):
        random.seed(seed)
        self.population = Population(1,num) 
        self.num = num
        self.generations = generations 
        self.generation = 0
        self.num_pop = num_pop
        self.verbose = verbose
        self.best_evaluation = 99999999
        self.mutation_rate = mutation_rate 
        self.best_evaluations = list()

        if(function == 'peaks'):
            self.function = Peaks(lbx = -3, ubx = 3,lby = -3, uby = 3)
        elif(function == 'eggholder'):
            self.function = Eggholder(lbx = -512, ubx = 512,lby = -512, uby = 512)
        elif(function == 'multimodal'):
            self.function = Multimodal(lbx = -3, ubx = 12.1,lby = -4.1, uby = 5.9)
        elif(function == 'crossintray'):
            self.function = CrossInTray(lbx = -10, ubx = 10,lby = -10, uby = 10)
        elif(function == 'rastrigin'):
            self.function = Rastrigin(lbx = -5.12, ubx = 5.12,lby = -5.12, uby = 5.12)
        else:
            self.function = Function(function+".txt")

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
            x = self.function.bin_to_real_x(bin = self.best_chromosome[:int(len(self.best_chromosome)/2)])
            y = self.function.bin_to_real_y(bin = self.best_chromosome[int(len(self.best_chromosome)/2):])
            if(self.verbose):
                print(f"best fitness: {self.best_evaluation} with x: {x} , y: {y} ")
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
        for i in self.measurements:
            x = self.function.bin_to_real_x(bin = self.measurements[i][:int(self.num/2)])
            y = self.function.bin_to_real_y(bin = self.measurements[i][int(self.num/2):])
            self.evaluations[i] = self.function.get_value(chromosome = self.measurements[i])
        
        min_index = min(self.evaluations.items(), key=itemgetter(1))[0]
        min_value = self.evaluations[min_index]

        if(min_value < self.best_evaluation):      
            self.best_chromosome = self.measurements[min_index]
            self.best_evaluation = self.evaluations[min_index]
    
    def update(self):

        for j in range(0,self.num):     # j index of the gene
            sum = 0
            for i in self.measurements.keys():
                sum += self.measurements[i][j]
            sum /= self.num_pop

            b = self.best_chromosome[j]

            if(b == 0 and sum >= 0.5):
                self.rotations[0][j] = (self.rotations[0][j] - self.delta) % (math.pi/2)
            elif (b == 1 and sum <= 0.5):
                self.rotations[0][j] = (self.rotations[0][j] + self.delta) % (math.pi/2)
    
    def mutation(self):
        for i in range(self.num):
            x = random.uniform(0,1)

            if(x < self.mutation_rate):
                self.rotations[0][i] *= -1