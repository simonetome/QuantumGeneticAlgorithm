<h2><b> Implementation of a Genetic Quantum Algorithm on a variational circuit
</b></h2>
</n>

This repository contains all the code used for the article and further experiments for constrained optimization (Knapsack).

</n>
<b>Repository structure:</b>

<pre>
.
├── GQA
│   ├── classical   /* Contains Genetic classical algorithms*/
│   │
│   ├── functions   /* Used functions for experiments*/
│   │   ├── crossintray.py
│   │   ├── eggholder.py
│   │   ├── function.py
│   │   ├── input_functions /* User defined */
│   │   │   └── func1.txt
│   │   ├── knapsack.py     /* For constrained optimization exp.*/
│   │   ├── multimodal.py
│   │   ├── peaks.py
│   │   └── rastrigin.py
│   ├── knapsack_instances
│   │   ├── instance1.txt 
│   │   ├── large_scale
│   │   └── low-dimensional
│   │       
│   ├── quantum /* Contains the GQA and classes used to simulate qubits behaviour*/
│   │   ├── compact.py  /* single chromosome variant for the GQA*/
│   │   ├── compact_knapsack.py
│   │   ├── compact_knapsack_qiskit.py
│   │   ├── gqa_function.py
│   │   ├── gqa_function_qiskit.py /* implementation using Qiskit library */
│   │   ├── gqa_knapsack.py
│   │   ├── gqa_qiskit.py
│   │   ├── population.py 
│   │   └── qbit.py
│   │
│   └── utils
│   
├── GuideNotebook_function.ipynb
├── GuideNotebook_knapsack.ipynb
├── classical_vs_quantum_function.py
├── experiments.py
├── experiments_knapsack.py
├── experiments_knapsack_thetas.py
├── experiments_mutation_crossover.py
├── exploration_exploitation.py
├── hyperparameters_benchmark_function.py
└── res.json
</pre>

Mapping between experiments reported in the article and used scripts.

<b> Tables 7 and 8: </b>
<pre>experiments_mutation_crossover.py</pre>
<b> Figure 8 and table 6: </b>
<pre> experiments.py</pre>
<b> Figures 9 to 11: </b>
<pre>exploration_exploitation.py</pre>

For an introdoction on how to use the defined algorithms, please refer to:
<pre>
├── GuideNotebook_function.ipynb
├── GuideNotebook_knapsack.ipynb
</pre>

<h3><b>Dependencies</b></h3>

Python 3.x.x : https://www.python.org/downloads/

To verify the installation, in Windows write in the terminal/cmd/powershell:
<pre>python</pre>
alternatively 
<pre>py</pre>
This should appear:
<pre>..informations about python version, day, etc.
>>> | </pre>


Used python libraries: Qiskit, Tqdm, Matplotlib, numpy. 

To install them, write in the terminal (after a succesful installation of python):
<pre>
pip install qiskit
pip install tqdm 
pip install matplotlib
pip install numpy
</pre>

Alternatively:

<pre>
pip3 install qiskit
pip3 install tqdm 
pip3 install matplotlib
pip3 install numpy
</pre>

<h2><b>Run</b></h2>

To run the scripts simply execute in the terminal/cmd/powershell:
<pre>py "script_name".py</pre>
