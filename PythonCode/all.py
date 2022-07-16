from cProfile import label
import abc_algo 
import whale_algo
import de_algo
import abc_de_algo
import abc_whale_algo
import abc_de_whale_algo
import objectives
import matplotlib.pyplot as plt
import threading
import time
import numpy
from data import ppp
from termcolor import colored
from math import log 
import sys

population=30
iter=100
limit=200
vardim=30
number_of_runs=1


obj_func=["Alpine","Dixonprice","Griewank","Qing","Rastrigin","Rosenbrock","Salomon","Sphere","Sumsquares","Trid","Weierstrass","Zakharov","Ackley"]
obj=obj_func[0]



print("Population size:",population)
print("Number of Iterations:",iter)
print("Dimensions:",vardim)
print("Objective function:",obj)
#c
if obj=="Griewank":
    ub=600
    lb=-600
elif obj=="Rosenbrock":
    ub=30
    lb=-30
elif obj=="Ackley":
    ub=32.768
    lb=-32.768
elif obj=="Rastrigin":
    ub=5.12
    lb=-5.12
elif obj=="Schwefel":
    ub=500
    lb=-500
elif obj=="Sphere":
    ub=5.12
    lb=-5.12
elif obj=="Weierstrass":
    ub=0.5
    lb=-0.5
elif obj=="Alpine":
    ub=10
    lb=-10
elif obj=="Dixonprice":
    ub=10
    lb=-10
elif obj=="Zakharov":
    ub=10
    lb=-5
elif obj=="Sumsquares":
    ub=10
    lb=-10
elif obj=="Trid":
    ub=vardim**2
    lb=-vardim**2
elif obj=="Salomon":
    ub=100
    lb=-100
elif obj=="Qing":
    ub=500
    lb=-500

else:
    print("Select correct objective function")
    sys.exit(0)


def f(pp,s,trace,name):
    if(pp):
        print(trace[-1])
        print(colored("Total time taken to execute is "+str(time.time()-s),'yellow'))
    print(colored(name,'red'))

def print_error_graph(errors):
    for error,c,algo in errors:
        temp=[log(d) for d in error[:-1] if d>0 ]
        plt.plot(temp,c,label=algo)
    plt.xlabel("Iteration")
    plt.ylabel("Log(error) value")
    plt.title("Hybrid algorithm for function optimization")
    plt.legend()
    plt.show()


def printResult(tt):
    errors=[]
    print("  Name             Best                Median              Worst               Mean                S.D             MeanError  "  )        

    for i in range(len(tt)):
        algo=tt[i][1]
        c=tt[i][2]
        tra=tt[i][0][0]
        error=[abs(i-tra[-1]) for i in tra]
        f=tt[i][0][1]
        foodScore=[]
        for i in f:
            i=(1-i)/i
            foodScore.append(i)
        print(algo,tra[-1],numpy.median(foodScore),tra[0],numpy.mean(foodScore),numpy.std(foodScore),numpy.mean(error))
        plt.plot([d for d in tra], c, label=algo)
        errors.append((error,c,algo))
    plt.xlabel("Iteration")
    plt.ylabel("function value")
    plt.title("Hybrid algorithm for function optimization")
    plt.legend()
    plt.show()
    plt.clf()
    print_error_graph(errors)

s=time.time()

def exe(func):
    global population,iter,limit,lb,ub,vardim,number_of_runs
    tt=[]
    ttt=[]
    for i in range(number_of_runs):
        dd=func(getattr(objectives,obj),population,iter,limit,lb,ub,vardim)
        tt.append(dd[0])
        ttt.append(dd[1])
    tt = numpy.mean(tt, axis=0,dtype=numpy.float128).tolist()
    ttt = numpy.mean(ttt, axis=0,dtype=numpy.float128).tolist()
    return tt,ttt
        
if(number_of_runs>1):
    trace=[]
    numpy.set_printoptions(precision=20)

    trace.append((exe(abc_algo.run),"ABC",'k'))
    f(ppp,s,trace,"ABC Completed\n")
    trace.append((exe(de_algo.run),"DE",'g'))
    f(ppp,s,trace,"DE Completed\n")
    trace.append((exe(whale_algo.run),"WHALE",'b'))
    f(ppp,s,trace,"WHALE Completed\n")
    trace.append((exe(abc_de_algo.run),"ABC_DE",'y'))
    f(ppp,s,trace,"ABC DE Completed\n")
    trace.append((exe(abc_whale_algo.run),"ABC_WHALE",'m'))
    f(ppp,s,trace,"ABC WHALE  Completed\n")
    trace.append((exe(abc_de_whale_algo.run),"ABC_DE_WHALE",'r'))
    f(ppp,s,trace,"ABC DE WHALE Completed\n")

else:
    trace=[]
    trace.append((abc_algo.run(getattr(objectives,obj),population,iter,limit,lb,ub,vardim),"ABC",'k'))
    f(ppp,s,trace,"ABC Completed\n")
    trace.append((de_algo.run(getattr(objectives,obj),population,iter,limit,lb,ub,vardim),"DE",'g'))
    f(ppp,s,trace,"DE Completed\n")
    trace.append((whale_algo.run(getattr(objectives,obj),population,iter,limit,lb,ub,vardim),"WHALE",'b'))
    f(ppp,s,trace,"WHALE Completed\n")
    trace.append((abc_de_algo.run(getattr(objectives,obj),population,iter,limit,lb,ub,vardim),"ABC_DE",'y'))
    f(ppp,s,trace,"ABC DE Completed\n")
    trace.append((abc_whale_algo.run(getattr(objectives,obj),population,iter,limit,lb,ub,vardim),"ABC_WHALE",'m'))
    f(ppp,s,trace,"ABC WHALE  Completed\n")
    trace.append((abc_de_whale_algo.run(getattr(objectives,obj),population,iter,limit,lb,ub,vardim),"ABC_WHALE_DE",'r'))
    f(ppp,s,trace,"ABC DE WHALE Completed\n")
e=time.time()
print(colored("Total time taken to execute is "+str(e-s),'yellow'))
printResult(trace)
