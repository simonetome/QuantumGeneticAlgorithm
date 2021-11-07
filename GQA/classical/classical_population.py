import random
import math

random.seed(1234)


# Class Population define the population for our algorithm. 
# Population has "num_pop" chromosomes where each one has "num" genes. A gene is a qBit. 

class ClassicalPopulation:

    def __init__(self,num_pop,num):
        self.num = num
        self.num_pop = num_pop
        # the population will be a dictionary: {"chromosome_index(int)" : <Qbit object>}
        self.chromosomes = {}
        self.random_initialization()

    # Initialize the population (done at each generation before anything else) where every gene is setted to state |0>

    def random_initialization(self):
        for i in range(0,self.num_pop):
            chromosome = []
            for j in range(0,self.num):
                chromosome.append(round(random.uniform(0, 1)))
            self.chromosomes[i] = chromosome
    
    def mutate(self,mutation_rate):
        for i in range(0,self.num_pop):
            for j in range(0,self.num):
                x = random.uniform(0, 1)
                if (x <= mutation_rate):
                    self.chromosomes[i][j] = int(not(self.chromosomes[i][j]))

        