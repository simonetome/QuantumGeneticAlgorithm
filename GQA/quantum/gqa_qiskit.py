
from operator import itemgetter
from GQA.functions.knapsack import Knapsack
import math 
import random
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister,circuit
from qiskit import *
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock.backends.manhattan import fake_manhattan
from qiskit.visualization import plot_histogram

from qiskit import IBMQ, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator

from math import pi
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy import exp,arange,absolute,sqrt,sin

from GQA.quantum.population import Population
from operator import itemgetter
from GQA.functions.knapsack import Knapsack
import math
import random 
from matplotlib import pyplot as plt




from qiskit.test.mock import FakeManhattan,FakeMontreal
fake_manhattan_backend = FakeManhattan()
sim_manhattan = AerSimulator.from_backend(fake_manhattan_backend, method= 'matrix_product_state')


class GQA_qiskit:

    def __init__(self,generations,num_pop,knapsack_definition,delta,penalty,seed,verbose,crossover_probability,mutation_rate,shots):
        random.seed(seed)
        
        self.generations = generations
        self.generation = 0
        
        self.shots = shots 

        self.num = 0
        self.num_pop = num_pop
        self.verbose = verbose
        self.rotations = {}
    

        self.knapsack = Knapsack(knapsack_definition,penalty)
        self.num = self.knapsack.size
        self.measurements = {}
        for i in range(0,num_pop):
            self.rotations[i] = [0 for j in range (0,self.num)]
            self.measurements[i] = []
          
        self.population = Population(num_pop = num_pop, num = self.num)
        self.delta = delta*(math.pi)
        self.best_fitness = 0
        self.best_chromosome = []
        self.evaluations = {}
        self.history = {}
        self.mutation_rate = mutation_rate
        self.best_evaluations = []
        self.crossover_probability = crossover_probability
        self.best_generation = 0
        self.best_weight = 0
        
        
        
        #self.backend_sim = Aer.get_backend('aer_simulator_matrix_product_state')

        # Build noise model from backend properties
        #TOKEN = "5e19a884422efbc1715dfb0f0df18f59e11159ddec73008e31a7fe1081e6b896fc7d973aa7d46ec32f66f4d66dc13a53cbc4580b7b15b6fe5e58e3a20435d98a"
        #IBMQ.save_account(TOKEN)
        # provider = IBMQ.load_account()
        # print(provider.backends())
        # self.backend = provider.get_backend('ibmq_rome')
        # self.noise_model = NoiseModel.from_backend(self.backend)

        # Get coupling map from backend
        # self.coupling_map = self.backend.configuration().coupling_map

        # # Get basis gates from noise model
        # self.basis_gates = self.noise_model.basis_gates


    # run the algorithm for every generation
    def run(self):
        while self.generation < self.generations:
            self.run_generation()
            self.generation += 1
            if(self.verbose):
                print(f"best fitness: {self.best_fitness}  with weight {self.knapsack.calculate_weight(self.best_chromosome)}")
            self.best_evaluations.append(self.best_fitness)

            #####################
            #print(self.evaluations)

    def plot_history(self):
        self.function.plot(self.best_history,self.generations)

    # Define single generation step, the order is self-explanatory
    def run_generation(self):

        # --------------- QUANTUM SUBROUTINE -----------------


        # Create a Quantum Circuit acting on the q register

        n = self.num_pop # number of chromosomes 
        m = self.num # number of genes


        qr = QuantumRegister(n*m, 'q')
        cr = ClassicalRegister(n*m, 'c')
        qc = QuantumCircuit(qr, cr)

        qc.h(qr[:])
        for c in self.rotations:
          for i in range(0,m):
            qc.ry(self.rotations[c][i],int(c))
      

        qc.barrier(qr)
        qc.measure(qr, cr)

        qc.draw()

        #job_sim = execute(qc, self.backend_sim, shots=1)

        # QISKIT --------------- DOC 
        # Transpile the circuit for the noisy basis gates
        # tcirc = transpile(circ, sim_vigo)

        # # Execute noisy simulation and get counts
        # result_noise = sim_vigo.run(tcirc).result()
        # counts_noise = result_noise.get_counts(0)
        # plot_histogram(counts_noise,
        #        title="Counts for 3-qubit GHZ state with device noise model")
        # fake_manhattan_backend = FakeManhattan()
        # sim_manhattan = AerSimulator.from_backend(fake_manhattan_backend, method= 'matrix_product_state')

        tcirc = transpile(qc,sim_manhattan)
        job_sim = sim_manhattan.run(tcirc,shots= self.shots).result()


        #job_sim = execute(qc,fake_manhattan,shots = 1).result()
        counts = job_sim.get_counts(qc)
        measurement = list(counts.keys())[0][::-1]

        for j in self.measurements:
          l = []
          for i in range(int(j),int(j)+self.num):
            l.append(int(measurement[i]))
          self.measurements[j] = l

        # -------------- CLASSICAL COMPUTATION ---------------

        self.evaluation()                                       # evaluate all the chromosomes 
        selected,non_selected = self.tournament_selection()     # select best chromosomes among the population
        self.update_theta_values()                              # update all the thetas for the next generation (values are accumulated)
        x = random.uniform(0, 1)
        if(x >= (1-self.crossover_probability)):
            self.crossover(selected, non_selected)              # crossover the selected chromosomes with single point technique
        self.mutation(mutation_probability=self.mutation_rate)



    # After the measurements we need to "repair" the chromosomes to respect the constraint of max_weight. 
    # If a chromosome violates the constraint is repaired (Knapsack class function) setting to 0 the genes which represent items with lowest profit over weight ratio
    # until the constraint is satisfied. This is a technique used in Integer linear programming. If a gene is "turned off" with the repairment then its rotation value is setted to 0 
    # and at the next generation we will have equal chance that this gene will be 0 or 1.
    def repair(self):

        for i in self.measurements: # i index of the chromosome
            repaired_chromosome, deleted_genes = self.knapsack.repairment(self.measurements[i])
            self.measurements[i] = repaired_chromosome[:]
            # reset rotations for deleted genes
            if(len(deleted_genes) > 0):
                for j in deleted_genes:
                    self.rotations[i][j] = 0

    # 1st operation
    # measure all the genes for all the chromosome and evaluate all the chromosomes. Update "evaluations{}" and "measurements{}"
    def evaluation(self):

        #################################
        #           REPAIR              #
        #################################
        self.repair()
        
    
        for i in self.measurements:
            self.evaluations[i] = self.knapsack.calculate_revenue(l = self.measurements[i])
        
        index = max(self.evaluations.items(), key=itemgetter(1))[0] # best chromosome index of the generation
    
        self.history[self.generation] = {}
        self.history[self.generation]['weight'] = self.knapsack.calculate_weight(l = self.measurements[index])
        self.history[self.generation]['profit'] = self.evaluations[index]

        # save the max_profit of the chromosome of that generation if no penalty would be applied 
        true_profits = []
        for i in self.measurements:
            true_profits.append(self.knapsack.calculate_revenue(l=self.measurements[i]))
        
        self.history[self.generation]['true_profit'] = max(true_profits)




    # 2nd operation
    # TODO: tournament selection (best way to explore the search space of solutions)
    # with such implementation half of the population is selected (the fittest ones)
    #
    # update:   best_fitness
    #           best_chromosome
    #
    # return:   selected        (dictionary with key = chromosome_index and value = fitness_value) : selected chromosomes
    #           non_selected    (dictionary with key = chromosome_index and value = fitness_value) : non selected chromosomes
    def selection(self):
        n = int(self.num_pop/2)
        
        selected = dict(sorted(self.evaluations.items(), key = itemgetter(1), reverse = True)[:n]) 
        non_selected = dict(sorted(self.evaluations.items(), key = itemgetter(1), reverse = True)[n:]) 
        best_key = max(selected, key=selected.get)
            
        if(self.generation == 0):
            self.best_chromosome = self.measurements[best_key][:]
            self.best_fitness = self.evaluations[best_key]
            self.best_evaluations.append(self.best_fitness)
            
        else:
            if(self.best_fitness < self.evaluations[best_key]):
                self.best_chromosome = self.measurements[best_key][:]
                self.best_fitness = self.evaluations[best_key]
                self.best_generation = self.best_generation
                self.best_weight = self.knapsack.calculate_weight(self.measurements[best_key][:])
                self.best_evaluations.append(self.best_fitness)
                

        return selected,non_selected
            
    def tournament_selection(self):
        
        selected = []
        non_selected = []
        chromosomes = [i for i in self.measurements.keys()]
        random.shuffle(chromosomes)

        for i in range(0,len(chromosomes),2):
            total = self.evaluations[chromosomes[i]] + self.evaluations[chromosomes[i+1]]
            if total == 0:
                total = 1
            p1 =  self.evaluations[chromosomes[i]] / total
            x = random.uniform(0,1)
            if(x > p1):
                selected.append(chromosomes[i+1])
                non_selected.append(chromosomes[i])
            else:
                selected.append(chromosomes[i])
                non_selected.append(chromosomes[i+1])
        
        best_key = max(self.evaluations, key=self.evaluations.get)
        if(self.generation == 0):
            self.best_chromosome = self.measurements[best_key][:]
            self.best_fitness = self.evaluations[best_key]
            
        else:
            if(self.best_fitness < self.evaluations[best_key]):
                self.best_chromosome = self.measurements[best_key][:]
                self.best_fitness = self.evaluations[best_key]
                self.best_generation = self.generation
                self.best_weight = self.knapsack.calculate_weight(self.best_chromosome)

        return selected,non_selected
            
    
    # 3rd operation
    # update rotations wrt to the simplified update table (based on the reference paper) b -> best chromosome
    #
    #   xj  ||  bj ||     delta_theta (how much to increment the rotation for next generation for gene j of chromosome i)
    #   0   ||  0  ||      0
    #   0   ||  1  ||      + delta
    #   1   ||  0  ||      - delta
    #   1   ||  1  ||      0


    def update_theta_values(self):

        # update rotation values for selected chromosomes and then perform crossover
        for i in self.measurements:  # i index of the chromosome
            for j in range(0,self.num):     # j index of the gene
                
                x = self.measurements[i][j]
                b = self.best_chromosome[j]

                update = 0
                
                if(x == 0 and b == 1):
                    update = self.delta

                elif(x == 1 and b == 0):
                    update = -self.delta
                
                self.rotations[i][j] += update
    
    # 4th operation
    # performing the crossover 
    # the crossover is single pointed and performed over the rotations (we cross the rotations not bits)
    # Parents are selected randomly and so couples are random (all the selected chromosome will create a crossover)
    # from the couple 2 new chromosome are created and the single point index is selected randomly
    #  
    # Example:
    #
    # |: single point index   
    #
    # p1 - xxxxxxx|oooooooooooooo
    # p2 - &&&&&&&|%%%%%%%%%%%%%%
    #
    # c1 - xxxxxxx|%%%%%%%%%%%%%%
    # c2 - &&&&&&&|oooooooooooooo
    def crossover(self,selected, non_selected):

        selected_indexes = [i for i in selected]
        non_selected_indexes = [i for i in non_selected]
        
        # random shuffle to create couples
        random.shuffle(selected_indexes)
        
        counter = 0

        for i in range(0,len(selected_indexes),2):

            point = random.randint(1, self.num -2)

            self.rotations[non_selected_indexes[counter]][:point] = self.rotations[selected_indexes[i]][:point]
            self.rotations[non_selected_indexes[counter]][point:] = self.rotations[selected_indexes[i+1]][point:]

            self.rotations[non_selected_indexes[counter+1]][:point] = self.rotations[selected_indexes[i+1]][:point]
            self.rotations[non_selected_indexes[counter+1]][point:] = self.rotations[selected_indexes[i]][point:]

            counter += 2
    
    def plot_history(self):
        x_generations = [i for i in range(0,self.generations)] 
        y_weights = []
        y_profits = []
        y_true_profits = []
        for k in self.history:
            y_weights.append(self.history[k]['weight'])
            y_profits.append(self.history[k]['profit'])
            y_true_profits.append(self.history[k]['true_profit'])
        
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(x_generations,y_profits,label='penalized profit')
        ax1.plot(x_generations,y_true_profits,label='profit')
        ax2.plot(x_generations,y_weights,label='weight')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Profit')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Weight')
        ax1.legend()
        ax2.legend()
        plt.show()


    def mutation(self,mutation_probability):

        for r in self.rotations:
            for i in range(0,self.num):
                x = random.uniform(0, 1)
                if(x <= mutation_probability):
                    self.rotations[r][i] *= -1


# import os
# abspath = os.path.join(os.getcwd(),"Code","GQA")
# print(abspath)
# file_name = "knapPI_3_100_1000_1"
# path = os.path.join(abspath,"knapsack_instances","large_scale",file_name)
# print(path)
# gqa = GQA_knapsack(generations=100,num_pop=100,knapsack_definition=path,delta=0.1,penalty=None,seed=123,verbose=1,
# crossover_probability=0.75,mutation_rate=0.01)
# gqa.run()
