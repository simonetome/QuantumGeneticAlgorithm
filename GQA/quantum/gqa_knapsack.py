from GQA.quantum.population import Population
from operator import itemgetter
from GQA.functions.knapsack import Knapsack
import math
import random 
from matplotlib import pyplot as plt



# Class GQA: define every step for the algorithm, from the quantum sub-routine and the classical part. Read the pdf for the specifications of the algorithm to get an overview of it (TODO)
# attributes:   generations     -> for how many generations the algorithm must run
#               generation      -> counter for the generation
#               population      -> Population object 
#               num             -> size of the chromosomes
#               num_pop         -> how many chromosomes for each pop
#               rotations       -> dictionary that saves the rotation to apply at each generation and its updated based on the evaluation (for each gene)
#               knapsack        -> knapsack instance 
#               max_weight      -> max_weight constraint for the knapsack problem 
#               delta           -> how much increment/decrement for the rotations
#               best_fitness    -> best fitness found yet
#               best_chromosome -> chromosome associated to the best fitness value 
#               evaluations     -> dict {chromosome_index : fitness_value} 
#               measurements    -> dict {chromosome_index : chromosome_measured -> (list "[1,0,0,...,1]")}
class GQA_knapsack:

    def __init__(self,generations,num_pop,knapsack_definition,delta,penalty,seed,verbose,crossover_probability,mutation_rate):
        random.seed(seed)
        
        self.generations = generations
        self.generation = 0
        
        self.num = 0
        self.num_pop = num_pop
        self.verbose = verbose
        self.rotations = {}
        self.knapsack = Knapsack(knapsack_definition,penalty)
        self.num = self.knapsack.size
        for i in range(0,num_pop):
            self.rotations[i] = [0 for j in range (0,self.num)]
        self.population = Population(num_pop = num_pop, num = self.num)
        self.delta = delta*(math.pi)
        self.best_fitness = 0
        self.best_chromosome = []
        self.evaluations = {}
        self.measurements = {}
        self.history = {}
        self.mutation_rate = mutation_rate
        self.best_evaluations = []
        self.crossover_probability = crossover_probability
        self.best_generation = 0
        self.best_weight = 0
    
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


    # Define single generation step, the order is self-explanatory
    def run_generation(self):

        # --------------- QUANTUM SUBROUTINE -----------------

        self.population.reset()                                 # reset all the population: all num_pop chromosomes are a sequence of |0> genes
        self.population.H_barrier()                             # all genes pass through an Hadamard gate to get into |+> state
        self.population.U_barrier(self.rotations)               # all genes are rotated with an Ry(theta) gate with a theta equal to the accumalated value in the classical computing routine.
        self.measurements = self.population.measure_barrier()   # measure everything
    
        # -------------- CLASSICAL COMPUTATION ---------------
        self.evaluation()                                      # evaluate all the chromosomes 
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
            
        else:
            if(self.best_fitness < self.evaluations[best_key]):
                self.best_chromosome = self.measurements[best_key][:]
                self.best_fitness = self.evaluations[best_key]
                self.best_generation = self.best_generation
                self.best_weight = self.knapsack.calculate_weight(self.measurements[best_key][:])
                

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
    
    # def plot_history(self):
    #     x_generations = [i for i in range(0,self.generations)] 
    #     y_weights = []
    #     y_profits = []
    #     y_true_profits = []
    #     for k in self.history:
    #         y_weights.append(self.history[k]['weight'])
    #         y_profits.append(self.history[k]['profit'])
    #         y_true_profits.append(self.history[k]['true_profit'])
        
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1,2,1)
    #     ax2 = fig.add_subplot(1, 2, 2)
    #     ax1.plot(x_generations,y_profits,label='penalized profit')
    #     ax1.plot(x_generations,y_true_profits,label='profit')
    #     ax2.plot(x_generations,y_weights,label='weight')
    #     ax1.set_xlabel('Generation')
    #     ax1.set_ylabel('Profit')
    #     ax2.set_xlabel('Generation')
    #     ax2.set_ylabel('Weight')
    #     ax1.legend()
    #     ax2.legend()
    #     plt.show()


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
