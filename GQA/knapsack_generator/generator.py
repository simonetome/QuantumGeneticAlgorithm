# generator of knapsack instances 

import random

seed = 1
random.seed(seed)

items = 10
instance = dict()
for i in range(items):
    instance[i] = {}
    instance[i]['profit'] = 0
    instance[i]['weight'] = 0

name = "generated_"+str(items)
