# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:14:22 2019

@author: Silvan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

#y(x)=a_0+sum_n a_n*cos(n*k*x)/n**d+b_n*sin(n*k*x)/n**d

def gen_coef(N,R_max,d):
    #generates 2*N random real numbers in [-R_max,R_max] in the form c=[[a_1,...,a_N/N^d],[b_1,...,b_N/N^d]]
    #d is a dampening factor which makes high frequency coefficients smaller
    return np.array([R_max*(2*np.random.rand(N)-1.0),R_max*(2*np.random.rand(N)-1.0)])/np.arange(1,N+1)**d

def path_2D(c,L,x):
    kf=2*np.pi/L                    #periodicity of fourier series
    N=len(c[0,:])
    y=np.zeros(len(x))
    for n in range(N):
        y+=c[0,n]*np.cos(n*kf*x)+c[1,n]*np.sin(n*kf*x)
    return y-np.sum(c[0,:])

def path_length(c,L,x,dx,dim):
    #c: fourier coefficient matrix
    #L: distance between starting and end point
    #x: discretized [0,L] array
    #dx: step size
    #dim: dimension, D=2,3
    
    kf=2*np.pi/L                    #periodicity of fourier series
    N=len(c[0,:])
    x+=dx/2.0           #shift evaluation points by dx/2 for better precision
    
    if(dim==2):
        dy_dx=np.zeros(len(x))
        for n in range(N):
            dy_dx+=n*kf*(-c[0,n]*np.sin(n*kf*x)+c[1,n]*np.cos(n*kf*x))
        return np.sum(np.sqrt(1+dy_dx[:-1]**2))*dx
    
    if(dim==3):
        dy_dx=np.zeros(len(x))
        dz_dx=np.zeros(len(x))
        cy=c[0]
        cz=c[1]
        for n in range(N):
            dy_dx+=n*kf*(-cy[0,n]*np.sin(n*kf*x)+cy[1,n]*np.cos(n*kf*x))
            dz_dx+=n*kf*(-cz[0,n]*np.sin(n*kf*x)+cz[1,n]*np.cos(n*kf*x))
        return np.sum(np.sqrt(1+dy_dx[:-1]**2+dz_dx[:-1]**2))*dx


def path_length_distribution(N,L,N_s,dim,N_c,R_max,damp):
    #N: Number of Paths
    #L: distance between starting and end point
    #N_s: Number of subdivisions of [0,L]
    #dim: Dimension, dimension=2,3
    #N_c: number of fourier coefficients
    #R_max: max value of fourier coefficient
    #damp: dampening
    
    x,dx,=np.linspace(0,L,N_s,retstep=True)
    
    s=np.empty(N)
    
    if(dim==2):
        for i in range(N):
            c=gen_coef(N_c,R_max,damp)
            s[i]=path_length(c,L,x,dx,2)
    
    if(dim==3):
        for i in range(N):
            c=np.array([gen_coef(N_c,R_max,damp),gen_coef(N_c,R_max,damp)])
            s[i]=path_length(c,L,x,dx,3)
            
    return s
        
def sum_over_paths(N,L,k,N_s,N_c,R_max,damp,dim):
    #N: Number of Paths
    #L: Distance between starting and end point
    #k: wavenumber of light
    #N_s: Number of subdivisions of [0,L]
    #N_c: Number of fourier coefficients
    #R_max: maximal value of fourier coefficient
    #damp: dampening of fourier coefficients
    #dim: dimension 
    t0=time.time()
    
    x,dx=np.linspace(0,L,N_s,retstep=True)
    
    K=0.0+0.0j
    
    index=int(N/100)
    
    K_prog=np.empty(100,dtype=complex)
    
    if(dim==2):
        n=0
        for i in range(N):
            c=gen_coef(N_c,R_max,damp)
            K+=np.exp(1j*k*path_length(c,L,x,dx,2))
            if(i%index == 0):
                K_prog[n]=K/(i+1)
                n+=1
                
    if(dim==3):
        n=0
        for i in range(N):
            c=np.array([gen_coef(N_c,R_max,damp),gen_coef(N_c,R_max,damp)])
            K+=np.exp(1j*k*path_length(c,L,x,dx,3))
            if(i%index == 0):
                K_prog[n]=K/(i+1)
                n+=1
        
    return K/N ,K_prog ,time.time()-t0

def sum_over_paths_con(N,L,k,N_s,N_c,R_max,damp,dim):
    #N: Number of Paths
    #L: Distance between starting and end point
    #k: wavenumber of light
    #N_s: Number of subdivisions of [0,L]
    #N_c: Number of fourier coefficients
    #R_max: maximal value of fourier coefficient
    #damp: dampening of fourier coefficients
    #dim: dimension 
    t0=time.time()
    
    x,dx=np.linspace(0,L,N_s,retstep=True)
    
    K=0.0
    
    index=int(N/100)
    
    K_prog=np.empty(100)
    
    if(dim==2):
        n=0
        for i in range(N):
            c=gen_coef(N_c,R_max,damp)
            K+=np.exp(-k*path_length(c,L,x,dx,2))
            if(i%index == 0):
                K_prog[n]=K/(i+1)
                n+=1
                
    if(dim==3):
        n=0
        for i in range(N):
            c=np.array([gen_coef(N_c,R_max,damp),gen_coef(N_c,R_max,damp)])
            K+=np.exp(-k*path_length(c,L,x,dx,3))
            if(i%index == 0):
                K_prog[n]=K/(i+1)
                n+=1        
    return K/N ,K_prog ,time.time()-t0

def Green(x,k,dim):
    if(dim==2):
        return 0.25j*scipy.special.hankel1(0,k*np.abs(x))
    if(dim==3):
        return np.exp(1j*k*x)/(4*np.pi*x)
        

def Green_con(x,k,dim):
    if(dim==2):
        return 0.25j*scipy.special.hankel1(0,1j*k*x)
    if(dim==3):
        return np.exp(-k*np.abs(x))/(4*np.pi*np.abs(x))
    
def SoP_Kernel(x,k,dim):
    N=len(x)
    K=np.empty(N,dtype=complex)
    if(dim==2):
        for i in range(N):
            K[i]=np.mean(sum_over_paths(20000,x[i],k,80,8,10*x[i],2,2)[1][50:])
    if(dim==3):
        for i in range(N):
            K[i]=np.mean(sum_over_paths(20000,x[i],k,80,8,10*x[i],2,3)[1][50:])
    return K
        

def Kernel(x,k,dim):
    if(dim==2):
        a0=0.1197
        b0=13.76
    if(dim==3):
        a0=0.0588
        b0=12.12
    return np.exp(1j*b0*k*x)*np.exp(-(k*x)**2/(4*a0))

def Kernel_con(x,k,dim):
    return np.mean(sum_over_paths_con(2*10**4,x,k,80,8,10*x,2,dim)[1][60:])

def Kernel_con_2(x,k,dim):
    if(dim==2):
        a0=0.1197
        b0=13.76
    if(dim==3):
        a0=0.0588
        b0=12.12
    return np.exp(-b0*k*x)*np.exp((k*x)**2/(4*a0))

def Helmholtz(f,r,k,dim):
    h=1E-5
    return (f(r+h)+f(r-h)-2*f(r))/h**2+(dim-1)/r*(f(r+h)-f(r-h))/(2*h)+k**2*f(r)




#plot paths
N=18
L=1.0
x=np.linspace(0,L,100)
for i in range(N):
    c=gen_coef(8,10*L,2)
    y=path_2D(c,L,x)
    plt.plot(x,y)
plt.savefig('presentation_cover4',dpi=200)
plt.show()




##helmholtz
#
#L=np.linspace(-1,1,100)
#
#H=Helmholtz(lambda x: Kernel(x,1.0,3),L,10,3)
#
#
#
#plt.plot(L,np.abs(H))
#plt.xlabel('L')
#plt.ylabel(r'$|(\Delta + k^2)K(k,L)|^2$')
##plt.text(0.3,1500,r'$|(\Delta + k^2)K(k,L)|^2$',fontsize=20)
#plt.savefig('delta',dpi=200)
#plt.show()




#N=10
#L=1.0
##x,dx=np.linspace(0,L,100,retstep=True)
#k=np.linspace(1,5,N)
#K=np.empty(N)
#for i in range(N):
#    print(i+1)
#    K[i]=Kernel_con(1.0,k[i],3)
#    
#p=np.polyfit(k,np.log(K/K[0]),1)
#
#plt.plot(k,np.log(K/K[0]),'ro',label='Simulation')
#plt.plot(k,np.polyval(p,k),'k-.',label='Fit')
#plt.plot(k,-L*(k-k[0]),'b',label=r'$-L(k-k^{\prime})$')
#plt.legend(loc='best')
#plt.show()





##comparison: pld kernel vs sum over paths
#N=15
#k=0.5
#L=np.linspace(0.01,1.7,N)
#Lc=np.linspace(L[0],L[-1],100)
#
#K_sum=np.empty(N,dtype=complex)
#K_sum_con=np.empty(N)
#
#t0=time.time()
#
#K_sum=SoP_Kernel(L,k,3)
#print(time.time()-t0)
##for i in range(N):
##    print(i+1)
##    K_sum[i]=Sop_Kernel(L[i],k,3)
##    K_sum_con[i]=np.mean(sum_over_paths_con(10000,L[i],k,80,8,10*L[i],2,3)[1][40:])
#    
##print('SoP:',time.time()-t0)
##
##t0=time.time()
##a=Kernel(L,k,2)
##b=Kernel_con_2(L,k,2)
##
##print('PLD:',time.time()-t0)
#
#plt.figure(1)
#plt.plot(L,np.real(K_sum),'ro',label='SoP Method')
#plt.plot(Lc,np.real(Kernel(Lc,k,3)),'b-.',label='PLD Method')
#plt.xlabel('L')
#plt.ylabel(r'$\Re \left[ K(L,k) \right]$')
#plt.legend(loc='best')
##plt.savefig('SoP_vs_PLD_real_k1',dpi=200)
#plt.show()
#
#plt.figure(2)
#plt.plot(L,np.imag(K_sum),'ro',label='SoP Method')
#plt.plot(Lc,np.imag(Kernel(Lc,k,3)),'b-.',label='PLD Method')
#plt.xlabel('L')
#plt.ylabel(r'$\Im \left[ K(L,k) \right]$')
#plt.legend(loc='best')
##plt.savefig('SoP_vs_PLD_imag_k1',dpi=200)
#plt.show()
#
##plt.figure(3)
##plt.plot(L,K_sum_con,'ro',label='SoP Method')
##plt.plot(Lc,Kernel_con_2(Lc,k,2),'b-.',label='PLD Method')
##plt.xlabel('L')
##plt.ylabel(r'$K(L,i k)$')
##plt.legend(loc='best')
###plt.savefig('SoP_vs_PLD_con_k1',dpi=200)
##plt.show()
    


##convergence of K
#N=10**4
#L=1.0
#k=1.0
#N_s=80
#Nf=8
#R=10*L
#damp=2
#dim=2
#
#K,K_prog,dt=sum_over_paths(N,L,k,N_s,Nf,R,damp,dim)
#
#plt.plot(np.abs(K_prog))
#plt.show()
#
#plt.plot(np.real(K_prog/np.abs(K_prog)))
#plt.show()





##convergence of analytically continued K
#N=5*10**3
#L=1.0
#k=0.1
#N_s=80
#Nf=8
#R=10*L
#damp=2
#dim=3
#
#K,K_prog,dt=sum_over_paths_con(N,L,k,N_s,Nf,R,damp,dim)
#
#plt.plot(K_prog)
#plt.show()




##comparison: analytically continued Green Function/Kernel
#L=np.arange(1,7)
#k=0.1
#
#K=np.empty(6)
#
#for i in range(6):
#    K[i]=Kernel_con(L[i],k,3)
#    
#G=Green_con(L,k,3)
#K2=Kernel_con_2(L,k,3)
#    
#plt.plot(L,K/K[0],'ro')
#plt.plot(L,K2/K2[0],'bo')
#plt.plot(L,G/G[0],'k')
#plt.show()





##path length distributions
#N_p=10000       #number of paths
#B=50            #number of bins
#
#L=1.0
#nf=8
#R=10*L
#damp=2
#
#pld1=path_length_distribution(N_p,1.0,80,2,nf,R,damp)
##counter1,edges1=np.histogram(pld1,B,normed=True)
##edges1=0.5*(edges1[:-1]+edges1[1:])
#
#pld2=path_length_distribution(N_p,2,80,2,nf,2*R,damp)
##counter2,edges2=np.histogram(pld2,B,normed=True)
##edges2=0.5*(edges2[:-1]+edges2[1:])
##
##def fit(x,a,b):
##    return np.sqrt(a/np.pi)*np.exp(-a*(x-b)**2)
##
##guess=np.array([0.1188/L**2,12.12*L])
##
##a1,b1=scipy.optimize.curve_fit(fit,edges1,counter1,guess)[0]
##a2,b2=scipy.optimize.curve_fit(fit,edges2,counter2,guess)[0]
#
#plt.hist(pld1,B,normed=True,label=r'$L=1$')
#plt.hist(pld2,B,normed=True,label=r'$L=2$')
#plt.xlabel('Path Length')
#plt.ylabel('Relative Frequency')
##plt.plot(edges1,fit(edges1,a1,b1),linewidth=2.0)
#plt.legend(loc='best')
##plt.savefig('pld_L_dependence',dpi=200)
#plt.show()
#
#print('a1,b1=',a1,b1)
#print('a2,b2=',a2,b2)
#print('a,b=', 0.5*(a1+a2),0.5*(b1+b2))





##path length distribution data analysis 2D
##dim=2 damp=2 R=10L
#L=np.array([0.5,1,2,3,4,6,8,10,15,20])
#a=np.array([0.479,0.1188,0.0299,0.0129,0.0074,0.00324,0.00186,0.00119,0.000531,0.000291])
#b=np.array([6.87,13.75,27.55,41.28,55.14,82.6,110.1,137.89,206.4,275.07])
#
#Lc=np.linspace(L[0],L[-1],100)
#
#plt.plot(L,a,'ro')
#plt.plot(Lc,0.1197/Lc**2,'-.')
#plt.xlabel(r'$L$')
#plt.ylabel(r'$a(L)$')
#plt.savefig('pld_2D_a',dpi=200)
#plt.show()
#
#plt.plot(L,b,'ro')
#plt.plot(Lc,13.76*Lc,'-.')
#plt.xlabel(r'$L$')
#plt.ylabel(r'$b(L)$')
#plt.savefig('pld_2D_b',dpi=200)
#plt.plot()




##path length distribution data analysis 3D
#L=np.array([1,2,3,5,8,10,15,20])
#a=np.array([0.0589,0.0145,0.0064,0.00231,0.000922,0.000557,0.00025,0.000144])
#b=np.array([12.12,24.26,36.27,60.5,96.9,121.1,181.9,242.24])
#
#Lc=np.linspace(L[0],L[-1],100)
#
#plt.plot(L,a,'ro')
#plt.plot(Lc,0.0588/Lc**2,'-.')
#plt.xlabel(r'$L$')
#plt.ylabel(r'$a(L)$')
#plt.savefig('pld_3D_a',dpi=200)
#plt.show()
#
#plt.plot(L,b,'ro')
#plt.plot(Lc,12.12*Lc,'-.')
#plt.xlabel(r'$L$')
#plt.ylabel(r'$b(L)$')
#plt.savefig('pld_3D_b',dpi=200)
#plt.plot()



def pld(x):
    #path length distribution for L=1,Nf=8,R_max=10,damp=2,dim=2
    return 0.19438*np.exp(-0.1187*(x-13.73)**2)





#def K(k,pld):
#    #k: wavenumber array
#    #pld: path length distribution function
#    out=np.empty(len(k),dtype=complex)
#    for i in range(len(k)):
#        R=scipy.integrate.quad(lambda x: pld(x)*np.cos(k[i]*x),1.0,np.infty)[0]
#        I=scipy.integrate.quad(lambda x: pld(x)*np.sin(k[i]*x),1.0,np.infty)[0]
#        out[i]=R+I*1.0j
#    return out





##example: comparison propagator/green's function and finding normalization
#
#def K(k):
#    return np.exp(-k**2/0.4748+13.73j*k)        
#
#k=np.linspace(0.1,2,50)
#L=1.0
#
#K=K(k)
#G=Green(1.0,k,2)
#
#'''
#def fit(x,a,b,c):
#    return a*np.exp(-b*(x-c)**2)
#
#a,b,c=scipy.optimize.curve_fit(fit,k[5:],np.abs(K[5:]/G[5:]),[2.5,3.0,0.34])[0]
#'''
#
#def N(k):
#    return 2.643*np.exp(-2.575*(k-0.255)**2)*np.exp(1j*13.0*k)*np.exp(-1j*np.pi/4)
#
##def N(k):
##    return 2.6*np.exp(-2.9*(k-0.3)**2)*np.exp(11.8j*k)
#
#plt.plot(k,np.abs(K),label='Propagator')
#plt.plot(k,np.abs(G),label='Green\'s function')
#plt.xlabel('Wavenumber k')
#plt.ylabel('Absolute Value')
#plt.title('L=1.0')
#plt.legend(loc='best')
#plt.show()
#
#
#
#plt.plot(k,np.abs(K),label='Normalized Propagator')
#plt.plot(k,np.abs(N(k)*G),label='Green\'s function')
#plt.xlabel('Wavenumber k')
#plt.ylabel('Absolute Value')
#plt.title('L=1.0')
#plt.legend(loc='best')
#plt.show()
#
#plt.plot(k,np.real(K),label='Normalized Propagator')
#plt.plot(k,np.real(N(k)*G),label='Green\'s function')
#plt.xlabel('Wavenumber k')
#plt.ylabel('Real part')
#plt.title('L=1.0')
#plt.legend(loc='best')
#plt.show()
#
#plt.plot(k,np.imag(K),label='Normalized Propagator')
#plt.plot(k,np.imag(N(k)*G),label='Green\'s function')
#plt.xlabel('Wavenumber k')
#plt.ylabel('Imaginary part')
#plt.title('L=1.0')
#plt.legend(loc='best')
#plt.show()





##comparison: Kernel/Green's Function
#k=1.0
#L=np.linspace(0.2,12.0,500)
#
##L=1.0
##k=np.linspace(0.1,2,100)
#
#K=Kernel(L,k,3)
#G=Green(L,k,3)
#
#plt.plot(L,np.real(K/K[0]),label='Kernel')
#plt.plot(L,np.real(G/G[0]),label="Green's Function")
#plt.xlabel('L')
#plt.ylabel('Real Part')
#plt.legend(loc='best')
#plt.savefig('Green_vs_Kernel_3D',dpi=200)
#plt.show()





##laplacian of Kernel
#k=1.0
#r=3.0
#x=np.linspace(-r,r,200)
#
#
#H=Helmholtz(lambda u: Kernel(u,k,2),x,k,3)
#
##plt.plot(x,np.real(H))
##plt.plot(x,np.imag(H))
#plt.plot(x,np.abs(H)**2)
#plt.xlabel('L')
#plt.ylabel(r'$\Im \left[ (\Delta + k^2)K(x,k) \right]$')
##plt.savefig('Helmholtz_Kernel_imag',dpi=200)
#plt.show()