import random
import math

random.seed(1234)

# Class to define a single qbit (with real amplitudes, no complex values yet)
# the l2_squared norm is 1 (with a margin of error in the order of 1e-6) 
# to don't break the quantum paradigm, after performing a measurement the qbit state collapse to |0> (alpha = 1, beta = 0) or |1>. 
# only useful Q-Gates are defined (in a future development it will be good to transfer them to utils and implement complex amplitudes)

const = 1/(math.sqrt(2))

class Qbit:

    # initialize amplitudes of the Qbit    
    def __init__(self,alpha,beta):
        self.alpha  =  alpha
        self.beta   =  beta
        # if (self.check_l2_norm() != 1):
        #     raise RuntimeError(f"Bad definition + {self.alpha**2 + self.beta**2}")

    def check_l2_norm(self):
        error = 1e-6
        return (self.alpha**2 + self.beta**2) > 1.0 - error and (self.alpha**2 + self.beta**2) < 1.0 + error  

    # apply H-Gate to the single qbit
    def hadamard(self):
        self.alpha = (self.alpha + self.beta)
        self.beta = (self.alpha - self.beta)
        self.alpha *= const
        self.beta *= const

    # apply a rotational gate to the qbit
    def rotation(self,theta):
        cos_ = math.cos(theta/2)
        sin_ = math.sin(theta/2)
        self.alpha = (self.alpha*cos_) + (self.beta*(-sin_))
        self.beta = (self.alpha*sin_) + (self.beta*cos_)
    
    # measure a qbit: collapse its state and return the bit-value
    def measure(self):
        x = random.uniform(0,1)
        if(x > self.alpha**2):
            self.alpha = 0
            self.beta = 1
            return 1
        else:
            self.alpha = 1
            self.beta = 0
            return 0
    
    def to_string(self):
        print(f"Qbit: alpha - {self.alpha} beta - {self.beta}")