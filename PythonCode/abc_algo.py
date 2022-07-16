import numpy as np
import random, math, copy
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
        
class ArtificialBeeSwarm:
    def __init__(self, obj,foodCount, onlookerCount, bound, maxIterCount=1000, maxInvalidCount=200):
        self.foodCount = foodCount                  
        self.onlookerCount = onlookerCount          
        self.bound = bound                          
        self.maxIterCount = maxIterCount            
        self.maxInvalidCount = maxInvalidCount 
        self.obj=obj     
        self.foodList = [ABSIndividual(self.bound,self.obj) for k in range(self.foodCount)]   
        self.foodScore = [d.score for d in self.foodList]                             
        self.bestFood = copy.copy(self.foodList[np.argmax(self.foodScore)])                  

    def updateFood(self, i):   
        k = random.randint(0, self.bound.shape[1] - 1)  
        j = random.choice([d for d in range(self.foodCount) if d !=i])   
        vi = copy.deepcopy(self.foodList[i])
        vi.chrom[k] += random.uniform(-1.0, 1.0) * (vi.chrom[k] - self.foodList[j].chrom[k]) 
        vi.chrom[k] = np.clip(vi.chrom[k], self.bound[0, k], self.bound[1, k])               
        vi.calculateFitness()
        if vi.score > self.foodList[i].score:          
            self.foodList[i] = vi
            if vi.score > self.foodScore[i]:           
                self.foodScore[i] = vi.score
                if vi.score > self.bestFood.score:     
                    self.bestFood = copy.copy(vi)
            self.foodList[i].invalidCount = 0
        else:
            self.foodList[i].invalidCount += 1
            
    def employedBeePhase(self):
        for i in range(0, self.foodCount):              
            self.updateFood(i)            

    def onlookerBeePhase(self): 
        maxScore = self.bestFood.score     
        accuFitness = [(0.9*d/maxScore+0.1, k) for k,d in enumerate(self.foodScore)]       
        for k in range(0, self.onlookerCount):
            i = random.choice([d[1] for d in accuFitness if d[0] >= random.random()]) 
            self.updateFood(i)

    def scoutBeePhase(self):
        def f():
            print(self.bestFood.score)
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
        
        #print(trace[0],trace[-1],np.mean(self.foodScore),np.median(self.foodScore),np.std(self.foodScore))
        if(pp):
            print("\n\n")
        return trace,self.foodScore

    

def run(obj,population,iter,limit,lb,ub,vardim):
    random.seed()
    bound = np.tile([[lb], [ub]], vardim)
    abs = ArtificialBeeSwarm(obj,population, population, bound, iter, limit)
    print(colored("ABC Started",'green'))

    return abs.solve()