import numpy as np
import random, math, copy
from numpy.random import rand
from numpy.random import choice
from numpy import clip
from data import color,endc,pp
from termcolor import colored
 


 
class ABSIndividual:
    def __init__(self, bound,obj):
        self.score = 0.
        self.invalidCount = 0
        self.obj=obj 
        self.chrom = [random.uniform(a,b) for a,b in zip(bound[0,:],bound[1,:])] 
        self.calculateFitness()        

    def calculateFitness(self): 
        self.score = self.obj(self.chrom)


def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[0,i], bounds[1,i]) for i in range(len(bounds[0]))]
    return mutated_bound


def crossover(mutated, target, dims, cr):
    p = rand(dims)
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


 
class ArtificialBeeSwarm:
    def __init__(self, obj,foodCount, onlookerCount, bound, maxIterCount=1000, maxInvalidCount=200):
        self.foodCount = foodCount                 
        self.onlookerCount = onlookerCount           
        self.bound = bound
        self.obj=obj
        self.F = 0.5  
        self.cr=0.7  
        self.maxIterCount = maxIterCount           
        self.maxInvalidCount = maxInvalidCount    
        self.foodList = [ABSIndividual(self.bound,self.obj) for k in range(self.foodCount)]
        self.foodScore = [d.score for d in self.foodList]  
        self.bestFood = copy.copy(self.foodList[np.argmax(self.foodScore)])
    

    def updateFood(self, i):         
        vj = copy.deepcopy(self.foodList[i]) 
        candidates = [candidate for candidate in range(self.foodCount) if candidate != i]
        ia,ib,ic=tuple(choice(candidates,3,replace=False))
        a=np.asarray(self.foodList[ia].chrom)
        b=np.asarray(self.foodList[ib].chrom)
        c=np.asarray(self.foodList[ic].chrom)
        mutated = mutation([a, b, c], self.F)
        mutated = check_bounds(mutated, self.bound)
        trial = crossover(mutated, self.foodList[i].chrom, len(self.bound[0]), self.cr)
        vj.chrom=trial
        vj.calculateFitness()
        if vj.score > self.foodList[i].score:           
            self.foodList[i] = vj
            if vj.score > self.foodScore[i]:            
                self.foodScore[i] = vj.score
                if vj.score > self.bestFood.score:     
                    self.bestFood = vj
            self.foodList[i].invalidCount = 0
        else:
            self.foodList[i].invalidCount += 1
            
    def employedBeePhase(self):
        for i in range(0, self.foodCount):              
            self.updateFood(i)            

    def onlookerBeePhase(self):
        foodScore = [d.score for d in self.foodList]  
        maxScore = np.max(foodScore)        
        accuFitness = [(0.9*d/maxScore+0.1, k) for k,d in enumerate(foodScore)] 
        for k in range(0, self.onlookerCount):
            i = random.choice([d[1] for d in accuFitness if d[0] >= random.random()]) 
            self.updateFood(i)

    def scoutBeePhase(self):
        for i in range(0, self.foodCount):
            if self.foodList[i].invalidCount > self.maxInvalidCount:                    
                self.foodList[i] = ABSIndividual(self.bound,self.obj)
                self.foodScore[i] = max(self.foodScore[i], self.foodList[i].score)

    def solve(self):
        trace = []
        tt=self.bestFood.score
        tt=(1-tt)/tt
        trace.append(tt)
        for k in range(self.maxIterCount):
            self.employedBeePhase()
            self.onlookerBeePhase()
            self.scoutBeePhase()
            tt=self.bestFood.score
            tt=(1-tt)/tt
            trace.append(tt)
            if(color and pp):
                print(colored("\r[At iteration "+str(k+1)+' the best fitness is '+str(tt)+"]",'blue',attrs=['bold']),end=endc)
            elif(pp):
                print("\r['At iteration '"+ str(k+1)+ "' the best fitness is '"+ str(tt)+"']",end=endc)
        if(pp):
            print("\n\n")
        return trace,self.foodScore

def run(obj,population,iter,limit,lb,ub,vardim):
    random.seed()
    bound = np.tile([[lb], [ub]], vardim)
    abs = ArtificialBeeSwarm(obj,population, population, bound, iter, limit)
    print(colored("ABC DE Started",'green'))
    return abs.solve()

 


 


 



