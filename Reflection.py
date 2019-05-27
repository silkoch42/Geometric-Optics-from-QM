# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:19:34 2019

@author: Silvan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

#k=2*np.pi/(635E-9)      #wavenumber for red light
k=200
#Position of Source
xa=1.0
ya=1.0

def dist(x,xb,yb):
    return np.sqrt(ya**2+(x+xa)**2)+np.sqrt(yb**2+(xb-x)**2)

def phase(x,xb,yb):
    return np.mod(k*dist(x,xb,yb),2*np.pi)

def A(xb,yb,L):
    R=scipy.integrate.quad(lambda x: np.cos(k*dist(x,xb,yb)),-L/2,L/2,epsrel=0.01,limit=1000)[0]
    I=scipy.integrate.quad(lambda x: np.sin(k*dist(x,xb,yb)),-L/2,L/2,epsrel=0.01,limit=1000)[0]
    return (R+1j*I)/L

def P(xb,yb,L):
    return np.abs(A(xb,yb,L))**2

def classical(x):
    a=0.037
    y=np.empty_like(x)
    for i in range(len(x)):
        if(x[i]<0.8):
            y[i]=0.0
        elif(x[i]>1.2):
            y[i]=0.0
        else:
            y[i]=a
    return y

N=100
L=0.2
xb=np.linspace(-L/2,L/2,N)
yb=ya

x=np.linspace(-L/2,L/2,N)

d=dist(x,xa,ya)

#Optical Path length 
plt.figure(1)
plt.plot(x,dist(x,xa,ya),label='A=(-1,1) \n B=(1,1)')
plt.title('Optical Path Lenght for different Reflection Points')
plt.xlabel('Reflection Point')
plt.ylabel('Optical Path Length')
plt.legend(loc='best')
#plt.savefig('Optical_Path',dpi=300)
plt.show()

#phase plot
plt.figure(2)
plt.plot(x,phase(x,xa,ya),label='A=(-1,1) \n B=(1,1) \n L=0.2 \n $\lambda$=0.001')
plt.title('Phase for different Reflection Points')
plt.xlabel('Reflection Point')
plt.ylabel('Phase in radians')
plt.legend(loc='best')
#plt.savefig('Phase',dpi=300)
plt.show()

xb=np.linspace(0,2.5,N)
yb=ya

sol=np.empty_like(xb)

for i in range(N):
    sol[i]=P(xb[i],yb,L)
    
plt.figure(3)
plt.plot(xb,sol/20)
#plt.plot(xb,classical(xb),label='Classical Expectation')
#plt.title('Probabilty Amplitude for a short, centered Mirror')
plt.xlabel('x coordinate of photomultiplier P')
plt.ylabel('Detection Probability')
plt.text(2,0.045,'S=(-1,1)')
plt.text(2,0.04,'P=(x,1)')
plt.text(2,0.035,'L=0.2')
plt.text(2,0.03,'k=200')
plt.legend(loc='best')
plt.savefig('reflection_low_k',dpi=300)
plt.show()