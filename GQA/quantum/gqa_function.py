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


# Class GQA: define every step for the algorithm, from the quantum sub-routine and the classical part. Read the pdf for the specifications of the algorithm to get an overview of it (TODO)
# attributes:   generations     -> for how many generations the algorithm must run
#               generation      -> counter for the generation
#               population      -> Population object 
#               num             -> size of the chromosomes
#               num_pop         -> how many chromosomes for each pop
#               rotations       -> dictionary that saves the rotation to apply at each generation and its updated based on the evaluation (for each gene)
#               peaks           -> peaks function 
#               delta           -> how much increment/decrement for the rotations
#               best_fitness    -> best fitness found yet
#               best_chromosome -> chromosome associated to the best fitness value 
#               evaluations     -> dict {chromosome_index : fitness_value} 
#               measurements    -> dict {chromosome_index : chromosome_measured -> (list "[1,0,0,...,1]")}
#               history         -> history of chromosomes for generation
#               best_history    -> best chromosomes for generation
class GQA_function:

    def __init__(self,generations,num,num_pop,delta,seed,verbose,function,decay,mutation_rate,crossover_probability):
        
        random.seed(seed)
        self.generations = generations
        self.generation = 0
        self.population = Population(num_pop = num_pop, num = num)
        self.num = num
        self.num_pop = num_pop
        self.verbose = verbose
        self.rotations = {}
        self.decay = decay
        for i in range(0,num_pop):
            self.rotations[i] = [0 for j in range (0,num)]

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
        self.delta = delta*(math.pi)
        self.best_fitness = +9999999999
        self.best_chromosome = []
        self.evaluations = {}
        self.measurements = {}
        self.history = [[] for i in range(0,self.generations)]
        self.best_history = []
        self.eval_history = {}
        self.best_generation = -1
        self.mutation_rate = mutation_rate
        self.best_evaluations = []
        self.crossover_probability = crossover_probability
    
    # run the algorithm for every generation
    def run(self):
        while self.generation < self.generations:
            self.run_generation()
            self.generation += 1
            x = self.function.bin_to_real_x(bin = self.best_chromosome[:int(len(self.best_chromosome)/2)])
            y = self.function.bin_to_real_y(bin = self.best_chromosome[int(len(self.best_chromosome)/2):])
            if(self.verbose):
                print(f"best fitness: {self.best_fitness} with x: {x} , y: {y} ")
            if(self.decay and self.generation>0):
                #self.delta = self.delta-((self.delta-0.01)/self.generations)*self.generation
                self.delta = (0.025 + ((self.delta/math.pi)-0.025)*(math.e**-(self.generation/40)))*math.pi
               # print(self.delta)
            self.best_evaluations.append(self.best_fitness)

    def plot_history(self):
        self.function.plot_contour_history(self.best_history,self.generations)

    # Define single generation step, the order is self-explanatory
    def run_generation(self):

        # --------------- QUANTUM SUBROUTINE -----------------

        self.population.reset()                                 # reset all the population: all num_pop chromosomes are a sequence of |0> genes
        self.population.H_barrier()                             # all genes pass through an Hadamard gate to get into |+> state
        self.population.U_barrier(self.rotations)               # all genes are rotated with an Ry(theta) gate with a theta equal to the accumalated value in the classical computing routine.
        self.measurements = self.population.measure_barrier()   # measure everything
    
        # -------------- CLASSICAL COMPUTATION ---------------

        self.evaluation()                                       # evaluate all the chromosomes 
        #selected,non_selected = self.tournament_selection()     # select best chromosomes among the population
        selected,non_selected = self.selection() 
        self.update_theta_values()                              # update all the thetas for the next generation (values are accumulated)
        x = random.uniform(0, 1)
        if(x >= (1-self.crossover_probability)):
            self.crossover(selected, non_selected)              # crossover the selected chromosomes with single point technique
        self.mutation(mutation_probability=self.mutation_rate)


    # 1st operation
    # measure all the genes for all the chromosome and evaluate all the chromosomes. Update "evaluations{}" and "measurements{}"
    def evaluation(self):
        
        self.measurements = self.population.measure_barrier()
        
        # repair not needed
        # self.repair()
        
        for i in self.measurements:
            x = self.function.bin_to_real_x(bin = self.measurements[i][:int(self.num/2)])
            y = self.function.bin_to_real_y(bin = self.measurements[i][int(self.num/2):])
            self.history[self.generation].append((x,y))
            self.evaluations[i] = self.function.get_value(chromosome = self.measurements[i])
        
        index = min(self.evaluations.items(), key=itemgetter(1))[0]
    
        x = self.function.bin_to_real_x(bin = self.measurements[index][:int(self.num/2)])
        y = self.function.bin_to_real_y(bin = self.measurements[index][int(self.num/2):])
        self.best_history.append((x,y))
        self.eval_history[self.generation] = self.function.get_value_from_reals(x, y)
            
        
        #print(f"history at generation {self.generation}: {self.history[self.generation]}")

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
        
        selected = dict(sorted(self.evaluations.items(), key = itemgetter(1), reverse = False)[:n]) 
        non_selected = dict(sorted(self.evaluations.items(), key = itemgetter(1), reverse = False)[n:]) 
        best_key = min(selected, key=selected.get)
            
        if(self.generation == 0):
            self.best_chromosome = self.measurements[best_key]
            self.best_fitness = self.evaluations[best_key]
        else:
            if(self.best_fitness > self.evaluations[best_key]):
                self.best_chromosome = self.measurements[best_key]
                self.best_fitness = self.evaluations[best_key]
                self.best_generation = self.generation
        return selected,non_selected
    

    def tournament_selection(self):
        
        selected = []
        non_selected = []
        chromosomes = [i for i in self.measurements.keys()]
        random.shuffle(chromosomes)

        for i in range(0,len(chromosomes),2):
            total = self.evaluations[chromosomes[i]] + self.evaluations[chromosomes[i+1]]
            if(total != 0):
                p1 =  self.evaluations[chromosomes[i]] / total
            else:
                p1 = 0.5
            x = random.uniform(0,1)
            if(x > p1):
                selected.append(chromosomes[i+1])
                non_selected.append(chromosomes[i])
            else:
                selected.append(chromosomes[i])
                non_selected.append(chromosomes[i+1])
        
        best_key = min(self.evaluations, key=self.evaluations.get)
        if(self.generation == 0):
            self.best_chromosome = self.measurements[best_key]
            self.best_fitness = self.evaluations[best_key]
        else:
            if(self.best_fitness > self.evaluations[best_key]):
                self.best_chromosome = self.measurements[best_key]
                self.best_fitness = self.evaluations[best_key]
                self.best_generation = self.generation

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

                
                f_x = self.evaluations[i]
                f_b = self.best_fitness

                if(x == 0 and b == 0):
                    update = 0 
                
                if(x == 0 and b == 1):
                    if(f_x >= f_b):
                        update = -self.delta
                    else:
                        update = self.delta

                if(x == 1 and b == 0):
                    if(f_x >= f_b):
                        update = self.delta
                    else:
                        update = -self.delta
                
                if(x == 1 and b == 1):
                    update = 0

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

    def mutation(self,mutation_probability):

        for r in self.rotations:
            for i in range(0,self.num):
                x = random.uniform(0, 1)
                if(x <= mutation_probability):
                    self.rotations[r][i] *= -1


    def plot_fitness(self):
        fig = plt.figure()

        x = range(self.generations)
        y = self.best_evaluations

        ax = fig.add_subplot(1,1,1)
        ax.plot(x,y,label=f"Best fitness")

        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.legend()

        plt.show()
