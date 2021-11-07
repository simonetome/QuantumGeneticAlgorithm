from GQA.classical.classical_population import ClassicalPopulation
from operator import itemgetter
from GQA.functions.knapsack import Knapsack
import math
import random 
from matplotlib import pyplot as plt




class GA_knapsack:

    def __init__(self,generations,num_pop,knapsack_definition,penalty,mutation_rate,seed,verbose,crossover_probability):
        random.seed(seed)
        self.generations = generations
        self.generation = 0
        self.num = 2
        self.num_pop = num_pop
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.knapsack = Knapsack(knapsack_definition,penalty)
        self.num = self.knapsack.size
        self.population = ClassicalPopulation(num_pop = num_pop, num = self.num)
        self.crossover_probability = crossover_probability
        self.best_fitness = 0
        self.best_chromosome = []
        self.evaluations = {}
        self.history = {}
        self.best_evaluations = []
        self.best_weight = 0
    
    # run the algorithm for every generation
    def run(self):
        self.population.random_initialization()
        while self.generation < self.generations:
            self.run_generation()
            self.generation += 1
            #print(f"best fitness: {self.best_fitness} with chromosome {self.best_chromosome} with weight {self.knapsack.calculate_weight(self.best_chromosome)}")
            if(self.verbose):
                print(f"best fitness: {self.best_fitness} with weight {self.knapsack.calculate_weight(self.best_chromosome)} real fitness: {self.knapsack.calculate_revenue(self.best_chromosome)}")
               # print(self.evaluations)
            self.best_evaluations.append(self.best_fitness)



    # Define single generation step, the order is self-explanatory
    def run_generation(self):
                         
        self.evaluation()                                       # evaluate all the chromosomes 
        selected,non_selected = self.tournament_selection()                # select best chromosomes among the population
        x = random.uniform(0, 1)
        if(x >= (1-self.crossover_probability)):
            self.crossover(selected, non_selected)                  # crossover the selected chromosomes with single point technique
        self.population.mutate(mutation_rate = self.mutation_rate)


    # 1st operation
    # measure all the genes for all the chromosome and evaluate all the chromosomes. Update "evaluations{}" and "measurements{}"
    def evaluation(self):
        
        #################################
        #           REPAIR              #
        #################################
        self.repair()
        
        for i in self.population.chromosomes:
            self.evaluations[i] = self.knapsack.calculate_revenue(l = self.population.chromosomes[i])
        
        index = max(self.evaluations.items(), key=itemgetter(1))[0] # best chromosome index of the generation
    
        self.history[self.generation] = {}
        self.history[self.generation]['weight'] = self.knapsack.calculate_weight(l = self.population.chromosomes[index])
        self.history[self.generation]['profit'] = self.evaluations[index]

        # save the max_profit of the chromosome of that generation if no penalty would be applied 
        true_profits = []
        for i in self.population.chromosomes:
            true_profits.append(self.knapsack.calculate_revenue_no_penalty(l=self.population.chromosomes[i]))
        
        self.history[self.generation]['true_profit'] = max(true_profits)

    def repair(self):

        for i in self.population.chromosomes: # i index of the chromosome
            repaired_chromosome, deleted_genes = self.knapsack.repairment(self.population.chromosomes[i])
            self.population.chromosomes[i] = repaired_chromosome[:]



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
            self.best_chromosome = self.population.chromosomes[best_key][:]
            self.best_fitness = self.evaluations[best_key]
        else:
            if(self.best_fitness < self.evaluations[best_key]):
                self.best_chromosome = self.population.chromosomes[best_key][:]
                self.best_fitness = self.evaluations[best_key]
                self.best_weight = self.knapsack.calculate_weight(self.best_chromosome)

        return selected,non_selected
            
    def tournament_selection(self):
        
        selected = []
        non_selected = []
        chromosomes = [i for i in self.population.chromosomes]
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
            self.best_chromosome = self.population.chromosomes[best_key][:]
            self.best_fitness = self.evaluations[best_key]
    
        else:
            if(self.best_fitness < self.evaluations[best_key]):
                self.best_chromosome = self.population.chromosomes[best_key][:]
                self.best_fitness = self.evaluations[best_key]
                self.best_generation = self.generation
                self.best_weight = self.knapsack.calculate_weight(self.best_chromosome)

        return selected,non_selected

    
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


            self.population.chromosomes[non_selected_indexes[counter]][:point] = self.population.chromosomes[selected_indexes[i]][:point]
            self.population.chromosomes[non_selected_indexes[counter]][point:] = self.population.chromosomes[selected_indexes[i+1]][point:]

            self.population.chromosomes[non_selected_indexes[counter+1]][:point] = self.population.chromosomes[selected_indexes[i+1]][:point]
            self.population.chromosomes[non_selected_indexes[counter+1]][point:] = self.population.chromosomes[selected_indexes[i]][point:]

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


# import os
# abspath = os.path.join(os.getcwd(),"Code","GQA")
# print(abspath)
# file_name = "knapPI_3_100_1000_1"
# path = os.path.join(abspath,"knapsack_instances","large_scale",file_name)
# print(path)
# ga = GA_knapsack(crossover_probability = 0.9,generations=200, num_pop=100, knapsack_definition=path, penalty=None, mutation_rate=0.05, seed=1, verbose=1)
# ga.run()
