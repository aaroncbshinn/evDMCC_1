#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 08:49:42 2025

@author: acshinn
"""



import time
import numpy as np
np.set_printoptions(precision=8, linewidth=200, suppress=True)
import math
import sys
import matplotlib.pyplot as plt
import csv
numpy_memory = 16


delta=104.255


# temperature (cm-1) 
temp = 2*delta
bet = 1/temp
# reorganization energy (cm-1)
reorge = delta*1
#number of spin states
nparts=2
nstates =4
# bath characteristic frequency (cm-1)
omegac = delta*1
oc = omegac
# number of bath oscillators per site
nvib = nstates
#number of modes available to each oscillator
nosc =1
# maximum bath frequency 20-200 \omega_c
omegamax = 20*omegac
# reduced Planck constant (cm-1*ps)
hbar = 5.034*1.05
#dissipation constant
gamma=delta*.5
#coupling constant (fermions to bosons)
c =.05*delta

#Pauli Matrices and raising/lowering operators
eye=np.eye(nparts)
sigz=np.array([[1,0],[0,-1]])
sigx=np.array([[0,1],[1,0]])
sigm=np.array([[0,1],[0,0]])
sigp=np.array([[0,0],[1,0]])

#routine to build correlated multi-body pauli operators
def kronton(m):
    bigx=[]
    for i in range(0,nstates):
        init0=[]
        for k in range(0,nstates):
            init0.append(eye)
        init0[i]=m
        start=init0[0]
        for j in range(1,nstates):
            start=np.kron(start, init0[j])
            #print(start)
        bigx.append(start)
        
    return bigx
#for the interaction hamiltonian
def ironton(m):
    bigx=[]
    for i in range(0,nvib):
        init0=[]
        for k in range(0,nvib):
            init0.append(np.eye(nosc))
        init0[i]=m
        start=init0[0]
        for j in range(1,nstates):
            start=np.kron(start, init0[j])
            #print(start)
        bigx.append(start)
        
    return bigx
#Bath-related Hamiltonian operators
def bronton(m):
    bigx=np.zeros((nosc**nvib, nosc**nvib))
    for i in range(0,nvib):
        init0=[]
        for k in range(0,nvib):
            init0.append(np.eye(nosc))
        init0[i]=m
        # print(init0)
        start=init0[0]
        for j in range(1,nvib):
            start=np.kron(start, init0[j])
            #print(start)
        #print(start)
        bigx=bigx+start
    return bigx


#Jordan-Wigner transformed multi-body operators
def jwm(f):
    bigp=np.zeros((nparts**nstates, nparts**nstates))
    init0=[]
    for k in range(0,f):
        init0.append(eye)
    for k in range(f,nstates):
        init0.append(sigz)
    init0[f]=sigm
    #print(init0)
    start=init0[0]
    for j in range(1,nstates):
        start=np.kron(start, init0[j])
    return start
def jwp(f):
    bigp=np.zeros((nparts**nstates, nparts**nstates))
    init0=[]
    for k in range(0,f):
        init0.append(eye)
    for k in range(f,nstates):
        init0.append(sigz)
    init0[f]=sigp
    #print(init0)
    start=init0[0]
    for j in range(1,nstates):
        start=np.kron(start, init0[j])
    return start

#commutators used in the Lindblad Master Equation
def comm(A,B):
    c=A@B-B@A
    return c

def acomm(A,B):
    c=A@B+B@A
    return c


#Lindblad Master Equation, to be integrated with RK2
def Lind(H,p,jplus,jminus):
    pdot=-1j*comm(H,p)
    for i in (0,nstates-1):
        f=0
        f=(-.5*acomm(jplus[i]@jminus[i], p)+1*jminus[i]@p@jplus[i])*(1-vf[i])
        f+=(-.5*acomm(jminus[i]@jplus[i], p)+1*jplus[i]@p@jminus[i])*vf[i]
        pdot+=2*gamma*f
    return pdot


#Begin building the System Hamiltonian
#List of J-W operators to iterate over
#dagger/nondagger defined wrt initial condition
jwd=[]
jwnd=[]
for i in range(0,nstates):
    jwd.append(jwm(i))
    jwnd.append(jwp(i))
Hs=np.zeros([nparts**nstates, nparts**nstates])
#The system Hamiltonian is an nstate site Holstein model
for i in range(0,nstates):
    for j in range(0,nstates):
        if i==j-1:
            Hs+=jwd[j]@jwnd[i]
        if j==i-1:
            Hs+=jwd[j]@jwnd[i]
# Hs+=jwd[0]@jwnd[0]*-1
# Hs+=jwd[3]@jwnd[3]*1
            
#Kronecker to size
Hs=np.kron(Hs*delta,np.eye(nosc**nvib))


'Oscillator regular H'
#In the polaron, one harmonic oscillator per site
oscarray = np.zeros(nosc)
for i in range(len(oscarray)):
    oscarray[i] = (i)*oc*hbar
print(np.eye(nosc)*oscarray)
bath=np.eye(nosc)*oscarray
hb=bronton(bath)
hb=np.kron(np.eye(nparts**nstates), hb)
print('hb',hb)






#Interaction Hamiltonian, just the ladders up and down
Hv = np.zeros([nosc, nosc])
for i in range(0, nosc-1):
    Hv[i, i+1] = np.sqrt(i+1)
    Hv[i+1, i] = np.sqrt(i+1)
    
Hint=ironton(Hv)
Hi=np.zeros([nparts**nstates*nosc**nvib, nparts**nstates*nosc**nvib])
for i in range(0,nstates):
    Hi+=np.kron(jwd[i]@jwnd[i], Hint[i])
    
 
Hint=Hi

# print('Hint')
# print(Hint.round(decimals=1))
# print(Hint.shape)


'Add them all up, multiplying the interaction Hamiltonian by the coupling constant'
bigH = Hs+hb+Hint*c


rho=np.zeros(nstates)
vb = np.zeros(nvib)
ub = np.zeros(nvib)
# for i in np.arange(0,nvib):
#     vb[i] = 1.0/(math.exp(oscarray[i]/temp)-1)
#     ub[i] = 1.0 + vb[i]

# Electronic thermal factors 
vf = np.zeros(nstates)
uf = np.zeros(nstates)
#moving in to the grand canonical ensemble, we need the chemical potential
mu=np.array([-gamma,0,0,gamma])
for i in np.arange(0,nstates):
    vf[i] = 1.0/(math.exp((Hs[i,i]-mu[i])/temp)+1.0)
    uf[i] = 1.0 - vf[i]
    vf[0]=0
    vf[-1]=1
'Initial Condition set'


"Thermal probabilities and partition function"
part = np.zeros(nosc)
for i in range(0, len(part)):
    part[i] = np.exp(-bet*hbar*oc*(i))
partfun = np.sum(part)
prob = np.zeros(nosc)
for i in range(len(prob)):
    prob[i] = part[i % nosc]/partfun
    
    
# print('prob')
# print(prob.round(decimals=3))

bp=[]
for i in range(0,nvib):
    bp.append(prob*np.eye(nosc))

rhob=bp[0]
for i in range(1, nvib):
    rhob=np.kron(rhob,bp[i])
    
print('rhob')
# print(rhob.round(decimals=3))
# print(rhob.shape)


off=np.array([[0,0],[0,1]])
on=np.array([[1,0], [0,0]])
mid=np.array([[0.5,0],[0,0.5]])
ic=[]
# 


rhos=np.kron(mid,mid)
rhos=np.kron(rhos,mid)
rhos=np.kron(rhos,mid)
# # print('rhoshape')
# print(rhos)


rho=np.kron(rhos,rhob)


# print('anticommute check')
# print(acomm(jwd[0], jwnd[1]))

  
hi=np.eye(nosc**nvib)
Lplus=[]
Lminus=[]
for i in range(0, nstates):
    Lplus.append(np.kron(jwd[i], (hi)))
    Lminus.append(np.kron(jwnd[i],(hi)))
#vf=np.array([.5,.5,.5,.5])

'initiate final arrays of time and probability'
t1 = np.linspace(0, .2, 1000)
Pt1 = np.zeros(len(t1))
Pt2 = np.zeros(len(t1))
Pt3 = np.zeros(len(t1))
Pt4 = np.zeros(len(t1))
P = np.zeros(len(t1))
dt=t1[1]-t1[0]



#observables
n=[]
for i in range(0,nstates):
    n.append(np.kron(jwd[i]@jwnd[i], np.eye(nosc**nstates)))
#print(bign)
curr=np.zeros([nparts**nstates*nosc**nvib,nparts**nstates*nosc**nvib])
for i in range(1,2):
    curr+=(Lplus[i]@Lminus[i+1]-Lplus[i+1]@Lminus[i])
current=np.zeros(len(t1))+0j


#Actual RK 2 Routine of integration
for i in range(len(t1)):
    psol=Lind(bigH,rho, Lplus,Lminus)+0j
    p_dt=rho+psol*dt
    psol2=Lind(bigH,p_dt, Lplus,Lminus)+0j
    rho=rho+dt*(psol+psol2)/2
    f=rho.round(decimals=1).real
    
    #Get the expectation values
    Pt1[i]=np.trace(n[0]@rho)
    Pt2[i]=np.trace(n[1]@rho)
    Pt3[i]=np.trace(n[2]@rho)
    Pt4[i]=np.trace(n[3]@rho)
    current[i]=np.trace(curr@rho).imag
    
    if i%40==0:
        print(f)
        # print('count:',i)
        #print(current[i])
    t1[i]=t1[i]*delta/np.pi*2
    # with open('medadp2cp5g2oxct_1.csv', 'a') as f:
    #     writer=csv.writer(f, delimiter=',')
    #     writer.writerow([t1[i], Pt1[i], Pt2[i],Pt3[i],Pt4[i], current[i]])
    sys.stdout.flush()
    




plt.plot(t1, Pt1, 'k--', label='site 1 exact')
plt.plot(t1, Pt2, 'r--', label='site 2 exact')
plt.plot(t1, Pt3, 'g--', label='site 3 exact')
plt.plot(t1, Pt4, 'b--', label='site 4 exact')
plt.legend()
plt.title('c=%.3f, n=%1i'%(c, nosc))
plt.ylim(0,1)
plt.xlim(0,20)
plt.show()


plt.plot(t1,current)
plt.ylim(-1,1)
plt.xlim(0,20)
plt.show()
