from numpy import mat
from GQA.quantum.population import Population
from operator import itemgetter
from GQA.functions.knapsack import Knapsack
import math
import random 
from tqdm import tqdm
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister,circuit
from qiskit import *
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock.backends.manhattan import fake_manhattan
from qiskit.visualization import plot_histogram

from qiskit import IBMQ, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel


import math
import random 
from tqdm import tqdm



from qiskit.test.mock import FakeManhattan,FakeMontreal
# Manhattan has 65 qubits
fake_manhattan = FakeManhattan()



class Compact_knapsack_qiskit:
    def __init__(self,generations,delta,num_pop,istance,verbose,mutation_rate,seed):
        random.seed(seed)
        

        self.generations = generations 
        self.generation = 0
        self.num_pop = num_pop
        self.verbose = verbose
        self.best_evaluation = -9999999999999999999
        self.mutation_rate = mutation_rate 

        self.knapsack = Knapsack(istance,penalty='quadratic')
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
            print(self.rotations)
            self.run_generation()
            self.generation += 1

            if(self.verbose):
                print(f"best fitness: {self.best_evaluation} ")
    
    def run_generation(self):
        # --------------- QUANTUM SUBROUTINE -----------------

        
        n = 1 # number of chromosomes 
        m = self.num # number of genes


        qr = QuantumRegister(n*m, 'q')
        cr = ClassicalRegister(n*m, 'c')
        qc = QuantumCircuit(qr, cr)

        qc.h(qr[:])
        
        for i in range(0,m):
            qc.ry(self.rotations[0][i],int(0))
      

        qc.barrier(qr)
        qc.measure(qr, cr)

        qc.draw()

        #job_sim = execute(qc, self.backend_sim, shots=1)
        
        job_sim = execute(qc,fake_manhattan,shots = self.num_pop).result()
        counts = job_sim.get_counts(qc)
        
        self.extract(counts)
        print(self.measurements)
        
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
        print(self.best_chromosome)
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
    

    # counts is a dict in the form {'001010': num} where num is the number of samples
    # extract fill self.measurements to fit data structures accordingly to the algorithm baseline
    def extract(self,counts): 
        counter = 0
        for k in counts:
            for i in range(counts[k]):
                self.measurements[counter] = list()
                for j in range(self.num):
                    self.measurements[counter].append(int(k[j]))
                counter += 1

    