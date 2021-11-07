from matplotlib import pyplot as plt 
from pylab import cm 

CMAP = cm.viridis
COLOR = 'black'
def plotList(l,params,generations):

    
    fig, ax = plt.subplots()

    for i in range(len(l)):
        
        x = range(generations)
        y = l[i]

        ax.plot(x,y,label=f"{params[i]}pi")

    ax.set_xlabel('Generation')
    ax.set_ylabel('f(x,y)')
    ax.legend()

    plt.show()

def plotComparison(l,generations):

    
    fig, ax = plt.subplots()

        
    x = range(generations)
    y = l[0]

    ax.plot(x,y,label=f"Classical")
    
    x = range(generations)
    y = l[1]

    ax.plot(x,y,label=f"Quantum")

    ax.set_xlabel('Generation')
    ax.set_ylabel('f(x,y)')
    ax.legend()

    plt.show()

def average_evals(l):
    
    generations = len(l[0])
    sum = []
    for i in range(generations):
        sum.append(0)

    for i in range(generations):
        for j in range(len(l)):
            sum[i] += l[j][i]
        sum[i] /= len(l)
    
    return sum

def plotTwoComparison(eval1,eval2,generations):

    
    fig, ax = plt.subplots()

        
    x = range(generations)
    y = eval1

    ax.plot(x,y,label=f"GQA")
    
    x = range(generations)
    y = eval2

    ax.plot(x,y,label=f"sGQA")

    ax.set_xlabel('Generation')
    ax.set_ylabel('fitness')
    ax.legend()

    plt.show()