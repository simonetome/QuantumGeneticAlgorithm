import json

# Knapsack. The instance is defined with knapsack_definition where we define items (index will define which gene represents it) weight and profit.
class Knapsack:
    def __init__(self,knapsack_definition,penalty):
        self.max_weight = 0
        self.penalty = penalty
        self.size = 1
        self.instance = self.instance_from_txt(knapsack_definition)
        

    def calculate_revenue(self,l):
        sum = 0
        penalty = 0
        for i in range(0,self.size):
            sum += (l[i] * self.instance[str(i)]["profit"])
        if (self.constraint_check(l) == 0):
            penalty = self.calculate_penalty(l)
        return sum-penalty



    def calculate_weight(self,l):
        sum = 0
        for i in range(0,self.size):
            sum += l[i] * self.instance[str(i)]["weight"]
        return sum


    def constraint_check(self,l):
        sum = 0
        for i in range(0,self.size):
            sum += l[i] * self.instance[str(i)]["weight"]
        return sum <= self.max_weight

    def repairment(self,l):
        profit_over_weight = {}
        profit_over_weight_sorted = {}
        measurement = l[:]
        deleted_indexes = []
        
        for i in range(0,len(l)):
            if(l[i] == 1):
                profit_over_weight [i] = self.instance[str(i)]["profit"]/self.instance[str(i)]["weight"]

        profit_over_weight_sorted = dict(sorted(profit_over_weight.items(), key=lambda item: item[1]))
        indexes = list(profit_over_weight_sorted.keys())
        
        counter = 0
        while(self.constraint_check(measurement) == 0):
            measurement[indexes[counter]] = 0
            deleted_indexes.append(indexes[counter])
            counter += 1
        
        #print(f"Measurement {l} repaired into {measurement} deleting {deleted_indexes}")
        return measurement,deleted_indexes

    def calculate_penalty(self,l):
        profit_over_weight = {}
        sum_ = 0
        sumx = 0    # total weight of selected items 
        
        for i in range(len(l)):
            profit_over_weight[i] = self.instance[str(i)]["profit"]/self.instance[str(i)]["weight"]
        
        for i in range(len(l)):
            if(l[i]):
                sumx += self.instance[str(i)]["weight"] # calculate sum of the weights of selected items 
        
        p = max(profit_over_weight.values()) # p = max profit/weight ratio 
        for i in range(len(l)):
            sum_ += (l[i]*self.instance[str(i)]['weight'])
        
        if (self.penalty == 'linear'):
            return  p*(sum_-self.max_weight)
        elif (self.penalty == 'quadratic'):
            return (p*(sum_-self.max_weight))**2
        else:
            return 0

    # define instance given the txt file:
    def instance_from_txt(self,instance_file):
        instance = {}
        instance_file = instance_file
        file1 = open(instance_file, 'r')
        Lines_ = file1.readlines()
        Lines = []
        count = 0
        # Strips the newline character
    

        for l in Lines_:
            Lines.append(l.replace("\n", ""))

        self.size = int(Lines[0].split(" ")[0])
        self.max_weight = float(Lines[0].split(" ")[1])

        counter = 0
        for l in Lines[1:-1]:
            instance[str(counter)] = {}
            instance[str(counter)]["profit"] = float(l.split(" ")[0])
            instance[str(counter)]["weight"] = float(l.split(" ")[1])
            counter+=1
        
        

        return instance

