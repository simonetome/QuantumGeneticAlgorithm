from operator import itemgetter

import math 
import random
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister,circuit
from qiskit import *
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock.backends.manhattan import fake_manhattan
from qiskit.visualization import plot_histogram

from qiskit import IBMQ, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator


n = 4 # number of chromosomes 
m = 5 # number of genes


qr = QuantumRegister(n*m, 'q')
cr = ClassicalRegister(n*m, 'c')
qc = QuantumCircuit(qr, cr)

qc.h(qr[:])
for c in range(n):
    for i in range(0,m):
        qc.ry(c,int(c*m+i))


qc.barrier(qr)
qc.measure(qr, cr)

print("Drawing...")
print(qc)
#qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
