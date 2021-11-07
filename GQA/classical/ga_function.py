from GQA.quantum.gqa_function import GQA_function
from GQA.classical.classical_population import ClassicalPopulation
from operator import itemgetter
from GQA.functions.peaks import Peaks
from GQA.functions.eggholder import Eggholder
from GQA.functions.multimodal import Multimodal
from GQA.functions.crossintray import CrossInTray
from GQA.functions.function import Function
import math
import random 


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
class GA_function:

    def __init__(self,generations,num,num_pop,mutation_rate,seed,verbose,function,crossover_probability):
        random.seed(seed)
        self.generations = generations
        self.generation = 0
        self.population = ClassicalPopulation(num_pop = num_pop, num = num)
        self.num = num
        self.num_pop = num_pop
        if(function == 'peaks'):
            self.function = Peaks(lbx = -3, ubx = 3,lby = -3, uby = 3)
        elif(function == 'eggholder'):
            self.function = Eggholder(lbx = -512, ubx = 512,lby = -512, uby = 512)
        elif(function == 'multimodal'):
            self.function = Multimodal(lbx = -3, ubx = 12.1,lby = -4.1, uby = 5.9)
        elif(function == 'crossintray'):
            self.function = CrossInTray(lbx = -10, ubx = 10,lby = -10, uby = 10)
        else:
            self.function = Function(function+".txt")
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.best_fitness = 9999999999
        self.best_chromosome = []
        self.evaluations = {}
        self.history = [[] for i in range(0,self.generations)]
        self.best_history = []
        self.eval_history = {}
        self.best_evaluations = []
        self.best_generation = 0
        self.crossover_probability = crossover_probability

    
    # run the algorithm for every generation
    def run(self):
        self.population.random_initialization()
        while self.generation < self.generations:
            self.run_generation()
            self.generation += 1
            x = self.function.bin_to_real_x(bin = self.best_chromosome[:int(len(self.best_chromosome)/2)])
            y = self.function.bin_to_real_y(bin = self.best_chromosome[int(len(self.best_chromosome)/2):])
            if(self.verbose):
                print(f"best fitness: {self.best_fitness} with x: {x} , y: {y} ")
            self.best_evaluations.append(self.best_fitness)

    def plot_history(self):
        self.function.plot(self.best_history,self.generations)

    # Define single generation step, the order is self-explanatory
    def run_generation(self):

        # -------------- CLASSICAL COMPUTATION ---------------

        self.evaluation()                                       # evaluate all the chromosomes 
        selected,non_selected = self.tournament_selection()                # select best chromosomes among the population
        x = random.uniform(0, 1)
        if (x >= (1-self.crossover_probability)):
            self.crossover(selected, non_selected)                  # crossover the selected chromosomes with single point technique
        self.population.mutate(mutation_rate = self.mutation_rate)



    # 1st operation
    # measure all the genes for all the chromosome and evaluate all the chromosomes. Update "evaluations{}" and "measurements{}"
    def evaluation(self):
        
        
        for i in self.population.chromosomes:
            x = self.function.bin_to_real_x(bin = self.population.chromosomes[i][:int(self.num/2)])
            y = self.function.bin_to_real_y(bin = self.population.chromosomes[i][int(self.num/2):])
            self.history[self.generation].append((x,y))
            self.evaluations[i] = self.function.get_value(chromosome = self.population.chromosomes[i])
        
        index = min(self.evaluations.items(), key=itemgetter(1))[0]
    
        x = self.function.bin_to_real_x(bin = self.population.chromosomes[index][:int(self.num/2)])
        y = self.function.bin_to_real_y(bin = self.population.chromosomes[index][int(self.num/2):])
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
        
        selected = dict(sorted(self.evaluations.items(), key = itemgetter(1), reverse = True)[:n]) 
        non_selected = dict(sorted(self.evaluations.items(), key = itemgetter(1), reverse = True)[n:]) 
        best_key = min(selected, key=selected.get)
            
        if(self.generation == 0):
            self.best_chromosome = self.population.chromosomes[best_key]
            self.best_fitness = self.evaluations[best_key]
          
        else:
            if(self.best_fitness > self.evaluations[best_key]):
                self.best_chromosome = self.population.chromosomes[best_key]
                self.best_fitness = self.evaluations[best_key]
                self.best_generation = self.generation

        return selected,non_selected
    
    def tournament_selection(self):
        """Tournament selection. Each chromosome fights against another random chromosome and its probability 
        of being selected is proportional to its fitness related to opponent fitness.

        Returns:
            selected list[int]: selected indexes
            non selected list[int]: non selected indexes
        """
        selected = []
        non_selected = []
        chromosomes = [i for i in self.population.chromosomes]
        random.shuffle(chromosomes)

        for i in range(0,len(chromosomes),2):
            total = self.evaluations[chromosomes[i]] + self.evaluations[chromosomes[i+1]]
            p1 =  self.evaluations[chromosomes[i]] / total
            x = random.uniform(0,1)
            if(x > p1):
                selected.append(chromosomes[i+1])
                non_selected.append(chromosomes[i])
            else:
                selected.append(chromosomes[i])
                non_selected.append(chromosomes[i+1])
        
        best_key = min(self.evaluations, key=self.evaluations.get)
       
        if(self.generation == 0):
            self.best_chromosome = self.population.chromosomes[best_key]
            self.best_fitness = self.evaluations[best_key]
        else:
            if(self.best_fitness > self.evaluations[best_key]):
                self.best_chromosome = self.population.chromosomes[best_key]
                self.best_fitness = self.evaluations[best_key]
                self.best_generation = self.generation

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
        """Crossover operation, acts on the rotation data structure and uses a single point approach.

        Args:
            selected ((list)[int]): [selected chromosomes indexes]
            non_selected ((list)[int]): [non selected chromosomes indexes]
        """
       
        selected_indexes = selected
        non_selected_indexes = non_selected
        
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


# Run the algorithm
