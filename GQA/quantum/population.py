
from GQA.quantum.qbit import Qbit
import math

# Class Population define the population for our algorithm. 
# Population has "num_pop" chromosomes where each one has "num" genes. A gene is a qBit. 


const = 1/(math.sqrt(2))

class Population:

    def __init__(self,num_pop,num):
        self.num = num
        self.num_pop = num_pop
        # the population will be a dictionary: {"chromosome_index(int)" : <Qbit object>}
        self.chromosomes = {}
        
    # this function applied to the Population will return a dictionary measured_chromosomes: {"chromosome_index(int)" : measured qbits of the chromosome -> list[1,0,...,1]}
    def measure_barrier(self):
        meausured_chromosomes = {}
        for i in self.chromosomes:
            genes = []
            for j in range(0,self.num):
                genes.append(self.chromosomes[i][j].measure())
            meausured_chromosomes[i] = genes
        return meausured_chromosomes
    
    def measure_compact(self):
        genes = []
        for j in range(0,self.num):
            genes.append(self.chromosomes[0][j].measure())
        return genes

    # Apply H-Gate to every single gene
    def H_barrier(self):
        for i in range(0,self.num_pop):
            for j in range(0,self.num):
                self.chromosomes[i][j].hadamard()

    # Apply a parametrized Rotational gate to every gene depending on the rotational value saved during the execution of the algorithm
    def U_barrier(self,theta_rotations):
        for i in range(0,self.num_pop):
            for j in range(0,self.num):
                self.chromosomes[i][j].rotation(theta_rotations[i][j])

    # Initialize the population (done at each generation before anything else) where every gene is setted to state |0>
    def reset(self):
        self.chromosomes = {}
        for i in range(0,self.num_pop):
            self.chromosomes[i] = [Qbit(1,0) for i in range (0,self.num)]
            

