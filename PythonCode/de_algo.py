from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
import math
from data import color,endc,pp
from termcolor import colored



def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound


def crossover(mutated, target, dims, cr):
    p = rand(dims)
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


def differential_evolution(obj,pop_size, bounds, iter, F, cr):
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    obj_all = [obj(ind) for ind in pop]
    best_obj = max(obj_all)
    prev_obj = best_obj
    obj_iter = list()
    for i in range(iter):
        if best_obj > prev_obj:
            prev_obj = best_obj
        tt=best_obj
        tt=(1-tt)/tt
        obj_iter.append(tt)
        for j in range(pop_size):
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            mutated = mutation([a, b, c], F)
            mutated = check_bounds(mutated, bounds)
            trial = crossover(mutated, pop[j], len(bounds), cr)
            obj_target = obj(pop[j])
            obj_trial = obj(trial)
            if obj_trial > obj_target:
                pop[j] = trial
                obj_all[j] = obj_trial
        best_obj = max(obj_all)
        if(color and pp):
            print(colored("\r[At iteration "+str(i+1)+' the best fitness is '+str(tt)+"]",'blue',attrs=['bold']),end=endc)
        elif(pp):
            print("\r['At iteration '"+ str(i+1)+ "' the best fitness is '"+ str(tt)+"']",end=endc)
    if(pp):
        print("\n\n")
    return obj_iter,obj_all


def run(obj,population,iter,limit,lb,ub,vardim):
    bounds = asarray([(lb, ub)]*vardim)
    F = 0.5
    cr = 0.7
    print(colored("DE Started",'green'))

    return differential_evolution(obj,population, bounds, iter, F, cr) 
