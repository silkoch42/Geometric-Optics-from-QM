# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:51:16 2019

@author: Silvan
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

k=1000

n1=2.0
n2=1.0

alpha=np.pi/6.0
beta=np.arcsin(n2/n1*np.sin(alpha))

ya=1.0
xa=-ya*np.tan(alpha)
yb=-1.0
xb=-yb*np.tan(beta)

def s(x):
    return n1*np.sqrt((xa-x)**2+ya**2)+n2*np.sqrt((xb-x)**2+yb**2)

def kernel(xa,xb):
    return 1.0/np.sqrt(xa**2+1)**(3/2.0)+1.0/np.sqrt(xa**2+1)**(3/2.0)



def K(R):
    L=1000     #Maximum Number of subdivisions for integral calculations
    eps=0.01
    N=50
    x,dx=np.linspace(0.01,R,N,retstep=True)
    real=np.empty(N)
    imag=np.empty(N)
    real[0]=scipy.integrate.quad(lambda x: np.cos(k*s(x)),-x[0],x[0],epsrel=eps,limit=L)[0]
    imag[0]=scipy.integrate.quad(lambda x: np.sin(k*s(x)),-x[0],x[0],epsrel=eps,limit=L)[0]
    for i in range(1,N):
        r1=scipy.integrate.quad(lambda x: np.cos(k*s(x)),-x[i]-dx,-x[i],epsrel=eps,limit=L)[0]
        r2=scipy.integrate.quad(lambda x: np.cos(k*s(x)),x[i],x[i]+dx,epsrel=eps,limit=L)[0]
        real[i]=real[i-1]+r1+r2
        i1=scipy.integrate.quad(lambda x: np.sin(k*s(x)),-x[i]-dx,-x[i],epsrel=eps,limit=L)[0]
        i2=scipy.integrate.quad(lambda x: np.sin(k*s(x)),x[i],x[i]+dx,epsrel=eps,limit=L)[0]
        imag[i]=imag[i-1]+i1+i2
    
    return np.sqrt(real**2+imag**2),x,real,imag



K2,x,r,i=K(3)
M=np.mean(K2[25:])

plt.plot(x,K2/M,label=r'$|\int_{-R}^{R}e^{i k s(x)}dx|^2$')
#plt.errorbar(x,K2/M,0.1*K2/M)
plt.xlabel(r'Integration range $R$')
plt.ylabel('Detection probabilty')
plt.legend(loc='best')
plt.text(2.4,0.2,r'$k=1000$')
#plt.text(1.1,0.5,r'$|\int_{-R}^{R}e^{i k s(x)}dx|^2$',fontsize=20)
plt.savefig('refraction_v3',dpi=200)
plt.show()

#N=20
#
#dx=np.linspace(0,10,N)
#
#P=np.ones(N)
#
#for i in range(N):
#    print(i+1)
#    P[i]=trans_amp(dx[i])
#
#
#plt.figure(1)
#plt.plot(dx,P/np.mean(P[20:]))
#plt.text(4.0,0.5,r'$|\int_{-\Delta x}^{\Delta x} e^{ik s(x)}dx$|',fontsize=20)
#plt.ylabel('Transition Amplitude')
#plt.xlabel(r'Integration Interval $ \Delta x$')
##plt.axis([0,10,0,1.1])
#plt.legend(loc='best')
##plt.savefig('refraction',dpi=200)
#plt.show()

#x=np.linspace(-5,5,100)
#
#plt.figure(2)
#plt.plot(x,s(x))
#plt.show()
#
#d=np.linspace(0,5,100)
#xa=-d/2
#xb=d/2
#plt.figure(3)
#plt.plot(d,kernel(xa,xb)**2)
#plt.show()