import math

import numpy as np


def Griewank(data):
    s1 = 0.
    s2 = 1.
    for k, x in enumerate(data):
        s1 = s1 + x ** 2
        s2 = s2 * math.cos(x/math.sqrt(k+1))
    y = (1./4000.) * s1-s2 + 1
    return 1./(1. + y)

def Rastrigin(data):
    s = 10. * 25.
    for i in data:
        s = s + ( (i ** 2) - (10 * math.cos(2 * math.pi * i)) )
    s= (10.0*len(data))+s
    return 1./(1.+s)

def Rosenbrock(data):
    s=0.
    for i in range(len(data)-1):
        s=s+(100.*math.pow(data[i+1]-math.pow(data[i],2),2) + math.pow(1-data[i],2))
    return 1./(1.+s)

def Ackley(data):  
    a=20.
    b=0.2
    c=2*math.pi
    d=len(data)
    sum1 = sum(map(lambda i:i*i , data))
    sum2 = sum(map(lambda i:math.cos(c*i), data))
    s= -a * math.exp(-b * (math.sqrt(sum1/d))) - math.exp(sum2/d) + a + math.exp(1)
    return 1./(1.+s)

def Schwefel(data):
    d=len(data)
    s=0.
    for i in data:
        s=s+(i*math.sin(math.sqrt(abs(i))))
    s=(418.9829*d)-s
    return 1./(1.+s)

def Sphere(data):
    s=0.
    for i in data:
        s=s+(i*i)
    return 1./(1.+s)

def Weierstrass(data):
    k=20
    a=0.5
    b=3.0
    s=0.
    s1 = 0.0
    for i in range(len(data)):
        val = 0.0
        for j in range(k):
            val += a ** j * math.cos(2.0 * math.pi * b ** j * (data[i] + 0.5))
        s1 += val
    s2 = 0.0
    for i in range(k):
        s2 += a ** i * math.cos(2 * math.pi * b ** i * 0.5)
    s=s1 - len(data) * s2
    return 1./(1.+s)


def Alpine(data):
    s=0.
    for i in data:
        s=s+abs((i*math.sin(i)) + (0.1* i))
    return 1.+(1.+s)

def Dixonprice(data):
    s1=(data[0]-1)*(data[0]-1)
    s2=0.
    for i in range(1,len(data)):
        s2=s2+(i+1)*math.pow((2*math.pow(data[i],2)-data[i-1]),2)
    s=s1+s2
    return 1./(1.+s)

def Zakharov(data):
    s1=0.
    s2=0.
    s3=0.
    for i in range(len(data)):
        s1=s1+math.pow(data[i],2)
        s1=s2+math.pow((0.5*(i+1)*data[i]),2)
        s3=s3+math.pow(0.5*(i+1)*data[i],4)
    s=s1+s2+s3
    return 1./(1.+s)

def Sumsquares(data):
    s=0.
    for i in range(len(data)):
        s=s+((i+1)*data[i]*data[i])
    return 1./(1.+s)

def Trid(data):
    s1=0.
    s2=0.
    for i in data:
        s1=s1+((i-1)**2)
    for i in range(1,len(data)):
        s2=s2+(data[i]-data[i-1])
    s=s1-s2
    return 1./(1.+s)


def Salomon(data):
    val=0.0
    for i in data:
        val=val+ (i*i)
    s= 1.0 - np.cos(2.0 * np.pi * np.sqrt(val)) + 0.1 * val
    return 1./(1.+s)

def Qing(data):
    s=0.
    for i in range(len(data)):
        s=s+( ( (data[i]**2) - (i+1)) ** 2)
    return 1./(1.+s)

def Forrester(data):
    s=0.
    for i in data:
        s=s+ (math.pow((6*i)-2,2) * math.sin((12*i) - 4))
    return 1./(1.+s)

    



