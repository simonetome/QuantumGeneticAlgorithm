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


n = 1 # number of chromosomes 
m = 3 # number of genes


qr = QuantumRegister(n*m, 'q')
cr = ClassicalRegister(n*m, 'c')
qc = QuantumCircuit(qr, cr)
rotations = [-math.pi/2,0,math.pi/2]
qc.h(qr[:])
for c in range(n):
    for i in range(0,m):
        qc.ry(rotations[m-i-1],int(c*m+i))


qc.barrier(qr)
qc.measure(qr, cr)

print("Drawing...")
print(qc)
#qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})

simulator = Aer.get_backend('aer_simulator')
qc = transpile(qc, simulator)

# Run and get counts
result = simulator.run(qc).result()
counts = result.get_counts(qc)

print(counts)

plot_histogram(counts, title='Bell-State counts')
