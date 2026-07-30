#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:45:30 2024

@author: acshinn
"""

__authors__   =  "Philip Shushkov"
__date__      = "2024-08-12"

import time
import numpy as np
np.set_printoptions(precision=8, linewidth=200, suppress=True)
import math
import sys
import matplotlib.pyplot as plt
import csv



# time3, rho = np.loadtxt(r'correctmed.csv', dtype=float,
#                         delimiter=',', usecols=(0, 1), unpack=True)
# time2, rho2 = np.loadtxt(r'new.csv', dtype=float,
#                           delimiter=',', usecols=(0, 1), unpack=True)

numpy_memory = 8

# Spin-boson Hamiltonian parameters
# number of spin states
nstates = 2
# matrix elements of the spin Hamiltonian (cm^-1)
delta = 104.255
# including bias and shift
Hs = np.array([[0.000,104.255],[104.255,0.000]])
# temperature (cm-1) 
temp = 2.0*delta
# reorganization energy (cm-1)
reorge = delta
# bath characteristic frequency (cm-1)
omegac = delta/4.0
# number of bath oscillators
nvib = 20
# maximum bath frequency 20-200 \omega_c
omegamax = 40.0*omegac
# reduced Planck constant (cm-1*ps)
hbar = 5.3091 
# Rabi frequency [ps-1] = [THz]
omegaR=2.0*delta/hbar


# Bath discretization from Wang, Song, Chandler, Miller
# QHO frequency [cm^-1]
omega = np.zeros(nvib)
for i in np.arange(omega.shape[0]):
   omega[i] = omegamax*((i+1)/nvib)**2  

# Spin-vibrational coupling [cm^-1]^[1.5]
reorge_approx=0.0
ck = np.zeros(nvib)
for i in np.arange(ck.shape[0]):
   ck[i] = omega[i]*4.0*math.sqrt(omega[i]*omegamax)/(nvib*math.pi) # w*2/\pi*(1/(N/2*sqrt(w_w_max)))
   ck[i] = ck[i]*0.5*reorge*(omega[i]*omegac)/(omega[i]**2+omegac**2) # J(w)=0.5*E_r*w*w_c/(w^2+w_c^2)
   ck[i] = math.sqrt(ck[i])
   reorge_approx += 2.0*(ck[i]/omega[i])**2

print('Approximate E_r = ',reorge_approx)
print('Exact E_r = ',reorge)






# Spin-boson Hamiltonian [cm^-1]
# Linear spin-vibrational interaction
Vpqk = np.zeros(nstates*nstates*nvib).reshape(nstates,nstates,nvib) 
for i in np.arange(0,nvib):
   Vpqk[0,0,i] =  ck[i]/math.sqrt(2.0*omega[i])
   Vpqk[1,1,i] = -ck[i]/math.sqrt(2.0*omega[i])

shift=0
for i in range(len(omega)):
    shift+=Vpqk[0,0,i]/(omega[i])
print('shift is',shift)


# Vibrational thermal factors
vb = np.zeros(nvib)
ub = np.zeros(nvib)
for i in np.arange(0,nvib):
    vb[i] = 1.0/(math.exp(omega[i]/temp)-1.0)
    ub[i] = 1.0 + vb[i]

# Electronic thermal factors 
vf = np.zeros(nstates)
uf = np.zeros(nstates)
for i in np.arange(0,nstates):
    vf[i] = 1.0/(math.exp(Hs[i,i]/temp)+1.0)
    uf[i] = 1.0 - vf[i]

# Spin-vibrational Hamiltonian matrix elements with thermal factors
# Hs_ab*(v_b-v_a)
Hsuv = np.zeros(nstates*nstates).reshape(nstates,nstates)
for i in np.arange(Hsuv.shape[0]):
    for j in np.arange(Hsuv.shape[1]):
        Hsuv[i,j] = Hs[i,j]*(vf[j]-vf[i])

# V_{ab,k}*(v_b-v_a)
Vuv = np.zeros(nstates*nstates*nvib).reshape(nstates,nstates,nvib)
for i in np.arange(Vuv.shape[0]):
    for j in np.arange(Vuv.shape[1]):
        for k in np.arange(Vuv.shape[2]):
            Vuv[i,j,k] = Vpqk[i,j,k]*(vf[j]-vf[i])

# V_{ii,k}*v_i
Vk = np.zeros(nvib)
for k in np.arange(Vpqk.shape[2]):
    for i in np.arange(Vpqk.shape[0]):
        Vk[k] += Vpqk[i,i,k]*vf[i]

# V_{ab,k}(u_k*u_a*v_b-v_k*u_b*v_a)
Vuvu = np.zeros(nstates*nstates*nvib).reshape(nstates,nstates,nvib)
for i in np.arange(Vuvu.shape[0]):
    for j in np.arange(Vuvu.shape[1]):
        for k in np.arange(Vuvu.shape[2]):
            Vuvu[i,j,k] = Vpqk[i,j,k]*(ub[k]*uf[i]*vf[j]-vb[k]*uf[j]*vf[i])

# V_{ab,k}*(u_a+v_k)
Vu_v = np.zeros(nstates*nstates*nvib).reshape(nstates,nstates,nvib)
for i in np.arange(Vu_v.shape[0]):
    for j in np.arange(Vu_v.shape[1]):
        for k in np.arange(Vu_v.shape[2]):
            Vu_v[i,j,k] = Vpqk[i,j,k]*(uf[i]+vb[k])

# V_{ab,k}*(u_k-u_a)
Vu_u = np.zeros(nstates*nstates*nvib).reshape(nstates,nstates,nvib)
for i in np.arange(Vu_u.shape[0]):
    for j in np.arange(Vu_u.shape[1]):
        for k in np.arange(Vu_u.shape[2]):
            Vu_u[i,j,k] = Vpqk[i,j,k]*(ub[k]-uf[i])

Vu_ab = np.zeros(nstates*nstates*nvib).reshape(nstates,nstates,nvib)
for i in np.arange(Vu_ab.shape[0]):
    for j in np.arange(Vu_ab.shape[1]):
        for k in np.arange(Vu_ab.shape[2]):
            Vu_ab[i,j,k] = Vpqk[i,j,k]*(uf[i]*vf[j]-uf[j]*uf[i])



# Initialize amplitudes
# Electronic amplitudes
TSN = np.zeros(nstates*nstates).reshape(nstates,nstates) + 0.0j
TSN[0,0] = uf[0] + 0.0j # (+1) state - population 1 donor
TSN[1,1] = -vf[1] + 0.0j # (-1) state - population 0 acceptor



TSNop = np.zeros(nstates*nstates).reshape(nstates,nstates) + 0.0j
TSNop[0,0] = uf[0] + 0.0j # (+1) state - population 1 donor
TSNop[1,1] = -vf[1] + 0.0j # (-1) state - population 0 acceptor

TNSop = np.zeros(nvib) + 0.0j
# for i in np.arange(nvib):
#     TNSop[i] = -ck[i]/(math.sqrt(2.0*omega[i])*omega[i]) + 0.0j
    
    
TSSop = np.zeros(nstates*nstates*nvib).reshape(nstates,nstates,nvib) + 0.0j




# Vibrational amplitudes
# One-phonon
TNS = np.zeros(nvib) + 0.0j
for i in np.arange(nvib):
    TNS[i] = -ck[i]/(math.sqrt(2.0*omega[i])*omega[i]) + 0.0j

# Two-phonon Goepert
TNG = np.zeros(nvib*nvib).reshape(nvib,nvib) + 0.0j
# Two-phonon Raman
TNR = np.zeros(nvib*nvib).reshape(nvib,nvib) + 0.0j


#Three Phonons


TNT0 = np.zeros(nvib*nvib*nvib).reshape(nvib,nvib, nvib) + 0.0j

TNT1=np.zeros(nvib*nvib*nvib).reshape(nvib,nvib,nvib)+0.0j


# Electron-vibrational amplitudes
# Singles electron - one-phonon
TSS = np.zeros(nstates*nstates*nvib).reshape(nstates,nstates,nvib) + 0.0j
# Singles electron - two-phonon Goepert
TSG = np.zeros(nstates*nstates*nvib*nvib).reshape(nstates,nstates,nvib,nvib) + 0.0j
# Singles electron - two-phonon Raman
TSR = np.zeros(nstates*nstates*nvib*nvib).reshape(nstates,nstates,nvib,nvib) + 0.0j
#Singles Electron, Three Phonons
TST0=np.zeros(nstates*nstates*nvib*nvib*nvib).reshape(nstates, nstates, nvib, nvib, nvib)+0.0j
TST1=np.zeros(nstates*nstates*nvib*nvib*nvib).reshape(nstates, nstates, nvib, nvib, nvib)+0.0j



#one phonon equations
# def build_rhs_tsnop(TSNop,TNSop,TSSop,Hs,Hsuv,Vpqk,Vuv):
#     TSSco = np.conj(TSSop)
#     TNSso = TNSop + np.conj(TNSop)
#     rhs_tsnop = np.einsum('ai,ib->ab',Hs,TSNop) - np.einsum('ai,ib->ab',TSNop,Hs) + Hsuv
#     rhs_tsnop += np.einsum('ail,ibl->ab',Vpqk,TSSop) + np.einsum('ail,bil->ab',Vpqk,TSSco)
#     rhs_tsnop -= np.einsum('bil,ail->ab',Vpqk,TSSop) + np.einsum('bil,ial->ab',Vpqk,TSSco)
#     rhs_tsnop += np.einsum('ail,ib,l->ab',Vpqk,TSNop,TNSso,optimize=True)
#     rhs_tsnop -= np.einsum('bil,ai,l->ab',Vpqk,TSNop,TNSso,optimize=True)
#     rhs_tsnop += np.einsum('abi,i->ab',Vuv,TNSso)
#     return rhs_tsnop

# def build_rhs_tnsop(TSNop,TNSop,TSSop,omega,Vpqk,Vk):
#     rhs_tnsop = np.einsum('i,i->i',omega,TNSop) + np.einsum('ijf,ij->f',Vpqk,TSNop) + Vk
#     return rhs_tnsop

# def build_rhs_tssop(TSNop,TNSop,TSSop,Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u,Vuv):
#     TNSso = TNSop + np.conj(TNSop)
#     rhs_tssop = np.einsum('ai,ibf->abf',Hs,TSSop) - np.einsum('bi,aif->abf',Hs,TSSop) + np.einsum('f,abf->abf',omega,TSSop) 
#     rhs_tssop += Vuvu 
#     rhs_tssop += np.einsum('aif,ib->abf',Vu_v,TSNop) 
#     rhs_tssop -= np.einsum('bif,ai->abf',Vu_u,TSNop)
#     rhs_tssop -= np.einsum('ijf,aj,ib->abf',Vpqk,TSNop,TSNop) 
#     rhs_tssop += np.einsum('ail,l,ibf->abf',Vpqk,TNSso,TSSop,optimize=True)
#     rhs_tssop -= np.einsum('ibl,l,aif->abf',Vpqk,TNSso,TSSop,optimize=True)
#     return rhs_tssop






# multiphonon equations
def build_rhs_tsn(TSN,TNS,TSS,Hs,Hsuv,Vpqk,Vuv):
    TSSc = np.conj(TSS)
    TNSs = TNS + np.conj(TNS)
    rhs_tsn = np.einsum('ai,ib->ab',Hs,TSN) - np.einsum('ai,ib->ab',TSN,Hs) + Hsuv
    rhs_tsn += np.einsum('ail,ibl->ab',Vpqk,TSS) + np.einsum('ail,bil->ab',Vpqk,TSSc)
    rhs_tsn -= np.einsum('bil,ail->ab',Vpqk,TSS) + np.einsum('bil,ial->ab',Vpqk,TSSc)
    rhs_tsn += np.einsum('ail,ib,l->ab',Vpqk,TSN,TNSs,optimize=True)
    rhs_tsn -= np.einsum('bil,ai,l->ab',Vpqk,TSN,TNSs,optimize=True)
    rhs_tsn += np.einsum('abi,i->ab',Vuv,TNSs)
    return rhs_tsn

def build_rhs_tns(TSN,TNS,TSS,omega,Vpqk,Vk):
    rhs_tns = np.einsum('i,i->i',omega,TNS) + np.einsum('ijf,ij->f',Vpqk,TSN) + Vk
    return rhs_tns

def build_rhs_tng(TNG,TSS,omega,Vpqk):
    rhs_tng = np.einsum('i,ij->ij',omega,TNG) + np.einsum('j,ij->ij',omega,TNG) 
    rhs_tng +=  np.einsum('efd,efc->cd',Vpqk,TSS, optimize=True)
    rhs_tng +=  np.einsum('efc,efd->cd',Vpqk,TSS, optimize=True)
    return rhs_tng

def build_rhs_tnr(TNR,TSS,omega,Vpqk):
    TSSc = np.conj(TSS)
    rhs_tnr = np.einsum('i,ij->ij',omega,TNR) - np.einsum('j,ij->ij',omega,TNR) 
    rhs_tnr -= np.einsum('efd,efc->cd',Vpqk,TSS, optimize=True)
    rhs_tnr += np.einsum('efc,fed->cd',Vpqk,TSSc, optimize=True)
    return rhs_tnr

def build_rhs_tss(TSN,TNS,TSS,TNG,TNR,TSG,TSR,Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u,Vuv):
    TNSs = TNS + np.conj(TNS)
    rhs_tss = np.einsum('ai,ibf->abf',Hs,TSS) - np.einsum('bi,aif->abf',Hs,TSS) + np.einsum('f,abf->abf',omega,TSS) 
    rhs_tss += Vuvu 
    rhs_tss += np.einsum('aif,ib->abf',Vu_v,TSN) 
    rhs_tss -= np.einsum('bif,ai->abf',Vu_u,TSN)
    
   # rhs_tss -= np.einsum('ijf,aj,ib->abf',Vpqk,TSN,TSN) 
    
    rhs_tss += np.einsum('ail,l,ibf->abf',Vpqk,TNSs,TSS,optimize=True)
    rhs_tss -= np.einsum('ibl,l,aif->abf',Vpqk,TNSs,TSS,optimize=True)
    
    rhs_tss += np.einsum('agk,gb,ck->abc',Vpqk,TSN,TNG,optimize=True)
    rhs_tss -= np.einsum('bhk,ah,ck->abc',Vpqk,TSN,TNG,optimize=True)
    
    rhs_tss += np.einsum('agk,gb,ck->abc',Vpqk,TSN,TNR,optimize=True)
    rhs_tss -= np.einsum('bhk,ah,ck->abc',Vpqk,TSN,TNR,optimize=True)
    
    rhs_tss += np.einsum('abk,ck->abc',Vuv,TNR,optimize=True)
    rhs_tss += np.einsum('abk,ck->abc',Vuv,TNG,optimize=True)
    
    rhs_tss += np.einsum('aek,ebck->abc',Vpqk,TSR, optimize=True)
    rhs_tss -= np.einsum('bfk,afck->abc',Vpqk,TSR, optimize=True)
    
    rhs_tss += np.einsum('aek,ebck->abc',Vpqk,TSG, optimize=True)
    rhs_tss -= np.einsum('bfk,afck->abc',Vpqk,TSG, optimize=True)
    return rhs_tss
    



def build_rhs_tsg(TSN,TNS,TSS,TNG,TNR,TSG,TSR,TNT0, TNT1, TST0,TST1,Hs,omega,Vpqk,Vu_v,Vu_u, Vu_ab):
    Tns = np.conj(TNS)
    rhs_TSDG = np.einsum('ai,ibfe->abfe',Hs,TSG) - np.einsum('bi,aife->abfe',Hs,TSG)
    rhs_TSDG += np.einsum('f,abfe->abfe',omega,TSG) + np.einsum('e,abfe->abfe',omega,TSG)
    
    rhs_TSDG += np.einsum('aed,ebc->abcd',Vu_v,TSS, optimize=True)
    rhs_TSDG += np.einsum('aec,ebd->abcd',Vu_v,TSS, optimize=True)
    
    rhs_TSDG -= np.einsum('bfd,afc->abcd',Vu_u,TSS, optimize=True)
    rhs_TSDG -= np.einsum('bfc,afd->abcd',Vu_u,TSS, optimize=True)
    
    
    rhs_TSDG -= np.einsum('ihf,ibe,ah->abef',Vpqk,TSS,TSN, optimize=True)
    rhs_TSDG -= np.einsum('ihe,ibf,ah->abef',Vpqk,TSS,TSN, optimize=True)
    rhs_TSDG -= np.einsum('gjf,aje,gb->abef',Vpqk,TSS,TSN, optimize=True)
    rhs_TSDG -= np.einsum('gje,ajf,gb->abef',Vpqk,TSS,TSN, optimize=True)
    
    
    
    rhs_TSDG += np.einsum('aik,ibfe,k->abef',Vpqk,TSG,TNS, optimize=True)
    rhs_TSDG -= np.einsum('bjk,ajfe,k->abef',Vpqk,TSG,TNS, optimize=True)
    
    
    rhs_TSDG += np.einsum('aik,ibfe,k->abef',Vpqk,TSG,Tns, optimize=True)
    rhs_TSDG -= np.einsum('bjk,ajfe,k->abef',Vpqk,TSG,Tns, optimize=True)
    
    
    rhs_TSDG += np.einsum('aik,ibf,ek->abef',Vpqk,TSS,TNG, optimize=True)
    rhs_TSDG += np.einsum('aik,ibe,fk->abef',Vpqk,TSS,TNG, optimize=True)
    rhs_TSDG -= np.einsum('bjk,ajf,ek->abef',Vpqk,TSS,TNG, optimize=True)
    rhs_TSDG -= np.einsum('bjk,aje,fk->abef',Vpqk,TSS,TNG, optimize=True)
    
    
    
    rhs_TSDG += np.einsum('aik,ibf,ek->abef',Vpqk,TSS,TNR, optimize=True)
    rhs_TSDG += np.einsum('aik,ibe,fk->abef',Vpqk,TSS,TNR, optimize=True)
    rhs_TSDG -= np.einsum('bjk,ajf,ek->abef',Vpqk,TSS,TNR, optimize=True)
    rhs_TSDG -= np.einsum('bjk,aje,fk->abef',Vpqk,TSS,TNR, optimize=True)
    
    #Three Phonons
    rhs_TSDG += 1 * np.einsum('abk,fek->abef',Vu_ab,TNT0, optimize=True)
    rhs_TSDG += 1 * np.einsum('abk,fek->abef',Vu_ab,TNT1, optimize=True)
    
    rhs_TSDG += 1 * np.einsum('agk,fek,gb->abef',Vpqk,TNT0,TSN, optimize=True)
    rhs_TSDG += 1 * np.einsum('agk,fek,gb->abef',Vpqk,TNT1,TSN, optimize=True)
    
    rhs_TSDG += 1 * np.einsum('aik,ibfek->abef',Vpqk,TST0, optimize=True)
    rhs_TSDG += 1 * np.einsum('aqk,qbfek->abef',Vpqk,TST1, optimize=True)
    
    rhs_TSDG += -1 * np.einsum('bhk,fek,ah->abef',Vpqk,TNT0,TSN, optimize=True)
    rhs_TSDG += -1 * np.einsum('bhk,fek,ah->abef',Vpqk,TNT1,TSN, optimize=True)
    
    rhs_TSDG += -1 * np.einsum('bjk,ajfek->abef',Vpqk,TST0, optimize=True)
    rhs_TSDG += -1 * np.einsum('bqk,aqfek->abef',Vpqk,TST1, optimize=True)
    
    
    return rhs_TSDG

def build_rhs_tsr(TSN,TNS,TSS,TNG,TNR,TSG,TSR,TNT0, TNT1, TST0,TST1,Hs,omega,Vpqk,Vu_v,Vu_u, Vu_ab):
    Tns = np.conj(TNS)
    Tng = np.conj(TNG)
    Tss = np.conj(TSS)
    rhs_TSDR = np.einsum('ai,ibfe->abfe',Hs,TSR) - np.einsum('bi,aife->abfe',Hs,TSR)
    rhs_TSDR += np.einsum('f,abfe->abfe',omega,TSR) - np.einsum('e,abfe->abfe',omega,TSR)
    
    
    
    rhs_TSDR += np.einsum('aif,ibe->abef',Vu_u,TSS, optimize=True)
    rhs_TSDR -= np.einsum('bjf,aje->abef',Vu_v,TSS, optimize=True)
    
    rhs_TSDR += np.einsum('aie,bif->abef',Vu_v,Tss, optimize=True)
    rhs_TSDR -= np.einsum('bje,jaf->abef',Vu_u,Tss, optimize=True)
    
    
    
    rhs_TSDR += np.einsum('ihf,ibe,ah->abef',Vpqk,TSS,TSN, optimize=True)
    rhs_TSDR += np.einsum('gjf,aje,gb->abef',Vpqk,TSS,TSN, optimize=True)
    
    rhs_TSDR -= np.einsum('ihe,bif,ah->abef',Vpqk,Tss,TSN, optimize=True)
    rhs_TSDR -= np.einsum('gje,jaf,gb->abef',Vpqk,Tss,TSN, optimize=True)
    
    
    
    rhs_TSDR += np.einsum('aik,ibef,k->abef',Vpqk,TSR,TNS, optimize=True)
    rhs_TSDR -= np.einsum('bjk,ajef,k->abef',Vpqk,TSR,TNS, optimize=True)
    
    
    rhs_TSDR += np.einsum('aik,ibef,k->abef',Vpqk,TSR,Tns, optimize=True)
    rhs_TSDR -= np.einsum('bjk,ajef,k->abef',Vpqk,TSR,Tns, optimize=True)
    
    
    rhs_TSDR += np.einsum('aik,ibe,kf->abef',Vpqk,TSS,TNR, optimize=True)
    rhs_TSDR -= np.einsum('bjk,aje,kf->abef',Vpqk,TSS,TNR, optimize=True)
    
    rhs_TSDR += np.einsum('aik,bif,ek->abef',Vpqk,Tss,TNR, optimize=True)
    rhs_TSDR -= np.einsum('bjk,jaf,ek->abef',Vpqk,Tss,TNR, optimize=True)
    
    
    rhs_TSDR += np.einsum('aik,bif,ek->abef',Vpqk,Tss,TNG, optimize=True)
    rhs_TSDR -= np.einsum('bjk,jaf,ek->abef',Vpqk,Tss,TNG, optimize=True)
    
    
    rhs_TSDR += np.einsum('aik,ibe,fk->abef',Vpqk,TSS,Tng, optimize=True)
    rhs_TSDR -= np.einsum('bjk,aje,fk->abef',Vpqk,TSS,Tng, optimize=True)
#Three Phonon

    rhs_TSDR += 1 * np.einsum('abk,efk->abef',Vu_ab,np.conj(TNT1), optimize=True)
    rhs_TSDR += 1 * np.einsum('abk,ekf->abef',Vu_ab,TNT1, optimize=True)
    
    
    rhs_TSDR += 1 * np.einsum('agk,efk,gb->abef',Vpqk,np.conj(TNT1),TSN, optimize=True)
    rhs_TSDR += 1 * np.einsum('agk,ekf,gb->abef',Vpqk,TNT1,TSN, optimize=True)
    
    
    rhs_TSDR += 1 * np.einsum('aik,ibefk->abef',Vpqk,np.conj(TST1), optimize=True)
    rhs_TSDR += 1 * np.einsum('aqk,qbekf->abef',Vpqk,TST1, optimize=True)
    
    rhs_TSDR += -1 * np.einsum('bhk,efk,ah->abef',Vpqk,np.conj(TNT1),TSN, optimize=True)
    rhs_TSDR += -1 * np.einsum('bhk,ekf,ah->abef',Vpqk,TNT1,TSN, optimize=True)
    
    
    rhs_TSDR += -1 * np.einsum('bjk,ajefk->abef',Vpqk,np.conj(TST1), optimize=True)
    rhs_TSDR += -1 * np.einsum('bqk,aqekf->abef',Vpqk,TST1, optimize=True) 
    return rhs_TSDR






def build_rhs_tnt0(TSN,TNS,TNG, TNR, TSS,TSG, TSR, TNT0, TNT1, TST0, TST1, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab):
    #rhs_TNT0=0
    rhs_TNT0 = 1 * np.einsum('f,def->def',omega,TNT0, optimize=True)
    rhs_TNT0 += 1 * np.einsum('e,def->def',omega,TNT0, optimize=True)
    rhs_TNT0 += 1 * np.einsum('d,def->def',omega,TNT0, optimize=True)
    
    rhs_TNT0 +=  1*np.einsum('ijf,ijed->def',Vpqk,TSG, optimize=True)
    rhs_TNT0 +=  1*np.einsum('ije,ijfd->def',Vpqk,TSG, optimize=True)
    rhs_TNT0 +=  1*np.einsum('ijd,ijfe->def',Vpqk,TSG, optimize=True)			
    #rhs_tnsd=TNSD
    return rhs_TNT0


def build_rhs_tnt1(TSN,TNS,TNG, TNR, TSS,TSG, TSR, TNT0, TNT1, TST0, TST1, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab):
    #rhs_TNT1=0
 
    rhs_TNT1 = 1 * np.einsum('e,def->def',omega,TNT1, optimize=True)
    rhs_TNT1 += 1 * np.einsum('d,def->def',omega,TNT1, optimize=True)
    rhs_TNT1 += -1 * np.einsum('f,def->def',omega,TNT1, optimize=True)
    
    rhs_TNT1 += 1* np.einsum('pqe,pqfd->def',Vpqk,TSR, optimize=True)
    rhs_TNT1 += 1 * np.einsum('pqd,pqfe->def',Vpqk,TSR, optimize=True)
    rhs_TNT1 += -1 * np.einsum('ijf,ijed->def',Vpqk,TSG, optimize=True)
    return rhs_TNT1

        
def build_rhs_tst0(TSN,TNS,TNG, TNR, TSS,TSR, TSG, TNT0, TNT1, TST0, TST1, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab):
    #rhs_TSTN=0
    rhs_TSTN = 1 * np.einsum('aq,qbdef->abdef',Hs,TST0, optimize=True)
    rhs_TSTN += -1 * np.einsum('bq,aqdef->abdef',Hs,TST0, optimize=True)
    
    rhs_TSTN += 1 * np.einsum('f,abdef->abdef',omega,TST0, optimize=True)
    rhs_TSTN += 1 * np.einsum('e,abdef->abdef',omega,TST0, optimize=True)
    rhs_TSTN += 1 * np.einsum('d,abdef->abdef',omega,TST0, optimize=True)
    
    rhs_TSTN += 1 * np.einsum('aif,ibed->abdef',Vu_v,TSG, optimize=True)
    rhs_TSTN += 1 * np.einsum('aie,ibfd->abdef',Vu_v,TSG, optimize=True)
    rhs_TSTN += 1 * np.einsum('aid,ibfe->abdef',Vu_v,TSG, optimize=True)
    
    rhs_TSTN += -1 * np.einsum('bjf,ajed->abdef',Vu_u,TSG, optimize=True)
    rhs_TSTN += -1 * np.einsum('bje,ajfd->abdef',Vu_u,TSG, optimize=True)
    rhs_TSTN += -1 * np.einsum('bjd,ajfe->abdef',Vu_u,TSG, optimize=True)
    
    
    rhs_TSTN += 1 * np.einsum('aqk,fek,qbd->abdef',Vpqk,TNT0,TSS, optimize=True)
    rhs_TSTN += 1 * np.einsum('aqk,fdk,qbe->abdef',Vpqk,TNT0,TSS, optimize=True)
    rhs_TSTN += 1 * np.einsum('aqk,edk,qbf->abdef',Vpqk,TNT0,TSS, optimize=True)
    
    rhs_TSTN += -1 * np.einsum('bqk,fek,aqd->abdef',Vpqk,TNT0,TSS, optimize=True)
    rhs_TSTN += -1 * np.einsum('bqk,fdk,aqe->abdef',Vpqk,TNT0,TSS, optimize=True)
    rhs_TSTN += -1 * np.einsum('bqk,edk,aqf->abdef',Vpqk,TNT0,TSS, optimize=True)
    
    
    rhs_TSTN += 1 * np.einsum('agk,fek,gbd->abdef',Vpqk,TNT1,TSS, optimize=True)
    rhs_TSTN += 1 * np.einsum('agk,fdk,gbe->abdef',Vpqk,TNT1,TSS, optimize=True)
    rhs_TSTN += 1 * np.einsum('agk,edk,gbf->abdef',Vpqk,TNT1,TSS, optimize=True)
    
    rhs_TSTN += -1 * np.einsum('bhk,fek,ahd->abdef',Vpqk,TNT1,TSS, optimize=True)
    rhs_TSTN += -1 * np.einsum('bhk,fdk,ahe->abdef',Vpqk,TNT1,TSS, optimize=True)
    rhs_TSTN += -1 * np.einsum('bhk,edk,ahf->abdef',Vpqk,TNT1,TSS, optimize=True)
    
    
    rhs_TSTN += 1 * np.einsum('aqk,qbdef,k->abdef',Vpqk,TST0,TNS, optimize=True)
    rhs_TSTN += -1 * np.einsum('bqk,aqdef,k->abdef',Vpqk,TST0,TNS, optimize=True)
    
    rhs_TSTN += 1 * np.einsum('aqg,qbdef,g->abdef',Vpqk,TST0,np.conj(TNS), optimize=True)
    rhs_TSTN += -1 * np.einsum('bqg,aqdef,g->abdef',Vpqk,TST0,np.conj(TNS), optimize=True)
    
    rhs_TSTN += 1 * np.einsum('aik,fk,ibed->abdef',Vpqk,TNG,TSG, optimize=True)
    rhs_TSTN += 1 * np.einsum('aik,ek,ibfd->abdef',Vpqk,TNG,TSG, optimize=True)
    rhs_TSTN += 1 * np.einsum('aik,dk,ibfe->abdef',Vpqk,TNG,TSG, optimize=True)
    
    rhs_TSTN += -1 * np.einsum('bjk,fk,ajed->abdef',Vpqk,TNG,TSG, optimize=True)
    rhs_TSTN += -1 * np.einsum('bjk,ek,ajfd->abdef',Vpqk,TNG,TSG, optimize=True)
    rhs_TSTN += -1 * np.einsum('bjk,dk,ajfe->abdef',Vpqk,TNG,TSG, optimize=True)
    
    rhs_TSTN += 1 * np.einsum('aik,fk,ibed->abdef',Vpqk,TNR,TSG, optimize=True)
    rhs_TSTN += 1 * np.einsum('aik,ek,ibfd->abdef',Vpqk,TNR,TSG, optimize=True)
    rhs_TSTN += 1 * np.einsum('aik,dk,ibfe->abdef',Vpqk,TNR,TSG, optimize=True)
    
    rhs_TSTN += -1 * np.einsum('bjk,fk,ajed->abdef',Vpqk,TNR,TSG, optimize=True)
    rhs_TSTN += -1 * np.einsum('bjk,ek,ajfd->abdef',Vpqk,TNR,TSG, optimize=True)
    rhs_TSTN += -1 * np.einsum('bjk,dk,ajfe->abdef',Vpqk,TNR,TSG, optimize=True)
    

    return rhs_TSTN


    
def build_rhs_tst1(TSN,TNS,TNG, TNR, TSS,TSR, TSG, TNT0, TNT1, TST0, TST1, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab):
    
    #rhs_TST1 = 0
    rhs_TST1 = 1 * np.einsum('aq,qbedf->abdef',Hs,TST1, optimize=True)
    rhs_TST1 += -1 * np.einsum('bq,aqedf->abdef',Hs,TST1, optimize=True)
    
    rhs_TST1 += 1 * np.einsum('e,abdef->abdef',omega,TST1, optimize=True)
    rhs_TST1 += 1 * np.einsum('d,abdef->abdef',omega,TST1, optimize=True)
    rhs_TST1 += -1 * np.einsum('f,abdef->abdef',omega,TST1, optimize=True)
    
    rhs_TST1 += 1 * np.einsum('aif,ibde->abdef',Vu_u,TSG, optimize=True)
    rhs_TST1 += -1 * np.einsum('bjf,ajde->abdef',Vu_v,TSG, optimize=True)
    
    rhs_TST1 += 1 * np.einsum('aqe,qbdf->abdef',Vu_v,TSR, optimize=True)
    rhs_TST1 += 1 * np.einsum('aqd,qbef->abdef',Vu_v,TSR, optimize=True)
    
    rhs_TST1 += -1 * np.einsum('bqe,aqdf->abdef',Vu_u,TSR, optimize=True)
    rhs_TST1 += -1 * np.einsum('bqd,aqef->abdef',Vu_u,TSR, optimize=True)
    
    rhs_TST1 += 1 * np.einsum('agk,edk,bgf->abdef',Vpqk,TNT0,np.conj(TSS), optimize=True)
    rhs_TST1 += -1 * np.einsum('bhk,edk,haf->abdef',Vpqk,TNT0,np.conj(TSS), optimize=True)
    
    rhs_TST1 += 1 * np.einsum('agk,ekf,gbd->abdef',Vpqk,TNT1,TSS, optimize=True)
    rhs_TST1 += 1 * np.einsum('agk,dkf,gbe->abdef',Vpqk,TNT1,TSS, optimize=True)
    
    rhs_TST1 += -1 * np.einsum('bhk,ekf,ahd->abdef',Vpqk,TNT1,TSS, optimize=True)
    rhs_TST1 += -1 * np.einsum('bhk,dkf,ahe->abdef',Vpqk,TNT1,TSS, optimize=True)
    
    rhs_TST1 += 1 * np.einsum('aqk,qbedf,k->abdef',Vpqk,TST1,TNS, optimize=True)
    rhs_TST1 += -1 * np.einsum('bqk,aqedf,k->abdef',Vpqk,TST1,TNS, optimize=True)
    
    rhs_TST1 += 1 * np.einsum('aqg,qbedf,g->abdef',Vpqk,TST1,np.conj(TNS), optimize=True)
    rhs_TST1 += -1 * np.einsum('bqg,aqedf,g->abdef',Vpqk,TST1,np.conj(TNS), optimize=True)
    
    rhs_TST1 += 1 * np.einsum('aqk,ek,qbdf->abdef',Vpqk,TNG,TSR, optimize=True)
    rhs_TST1 += 1 * np.einsum('aqk,dk,qbef->abdef',Vpqk,TNG,TSR, optimize=True)
    rhs_TST1 += -1 * np.einsum('bqk,ek,aqdf->abdef',Vpqk,TNG,TSR, optimize=True)
    rhs_TST1 += -1 * np.einsum('bqk,dk,aqef->abdef',Vpqk,TNG,TSR, optimize=True)
    
    rhs_TST1 += 1 * np.einsum('aik,fk,ibed->abdef',Vpqk,np.conj(TNG),TSG, optimize=True)
    rhs_TST1 += -1 * np.einsum('bjk,fk,ajed->abdef',Vpqk,np.conj(TNG),TSG, optimize=True)
    
    rhs_TST1 += 1 * np.einsum('aik,kf,ibed->abdef',Vpqk,TNR,TSG, optimize=True)
    rhs_TST1 += -1 * np.einsum('bjk,kf,ajed->abdef',Vpqk,TNR,TSG, optimize=True)
    
    rhs_TST1 += 1 * np.einsum('aqk,ek,qbdf->abdef',Vpqk,TNR,TSR, optimize=True)
    rhs_TST1 += 1 * np.einsum('aqk,dk,qbef->abdef',Vpqk,TNR,TSR, optimize=True)
    
    
    rhs_TST1 += -1 * np.einsum('bqk,ek,aqdf->abdef',Vpqk,TNR,TSR, optimize=True)
    rhs_TST1 += -1 * np.einsum('bqk,dk,aqef->abdef',Vpqk,TNR,TSR, optimize=True)
    				
    return rhs_TST1




# Iterations of time integration
print('\nStart Electron-Phonon DMCC iterations')
# timestep (ps)
dt = 0.001*2.0*math.pi/omegaR
# number of iterations - 20 Rabi periods
maxiter = 20 * 400
pt=np.zeros(maxiter)
pto=np.zeros(maxiter)
t=np.zeros(maxiter)
# time (ps) 
time = 0.0

# imaginary unit (-i)
eye = 0.0 - 1.0j
# (-i)/\hbar
eye = eye/hbar

for epdmcc_iter in range(1, maxiter + 1):
    # RK-2 integration method
    # slope at t
    # rhs_TSNop = eye * build_rhs_tsnop(TSNop,TNSop,TSSop,Hs,Hsuv,Vpqk,Vuv)
    # rhs_TNSop = eye * build_rhs_tnsop(TSNop,TNSop,TSSop,omega,Vpqk,Vk)
    # rhs_TSSop = eye * build_rhs_tssop(TSNop,TNSop,TSSop,Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u,Vuv)
    
    
    
    rhs_TSN = eye * build_rhs_tsn(TSN,TNS,TSS,Hs,Hsuv,Vpqk,Vuv)
    
    
    rhs_TNS = eye * build_rhs_tns(TSN,TNS,TSS,omega,Vpqk,Vk)
    rhs_TNG = eye * build_rhs_tng(TNG,TSS,omega,Vpqk)
    rhs_TNR = eye * build_rhs_tnr(TNR,TSS,omega,Vpqk)
    
    rhs_TSS = eye * build_rhs_tss(TSN,TNS,TSS,TNG,TNR,TSG,TSR, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u,Vuv)
    
    
    rhs_TSG = eye * build_rhs_tsg(TSN,TNS,TSS,TNG,TNR,TSG,TSR,TNT0, TNT1, TST0, TST1,Hs,omega,Vpqk,Vu_v,Vu_u, Vu_ab)
    rhs_TSR = eye * build_rhs_tsr(TSN,TNS,TSS,TNG,TNR,TSG,TSR,TNT0, TNT1, TST0, TST1,Hs,omega,Vpqk,Vu_v,Vu_u, Vu_ab)
    
    rhs_TNT0= eye * build_rhs_tnt0(TSN,TNS,TNG, TNR, TSS,TSG, TSR, TNT0, TNT1, TST0, TST1, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab)
    rhs_TNT1= eye * build_rhs_tnt1(TSN,TNS,TNG, TNR, TSS,TSG, TSR, TNT0, TNT1, TST0, TST1, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab)
    
    rhs_TST0= eye * build_rhs_tst0(TSN,TNS,TNG, TNR, TSS,TSR, TSG, TNT0, TNT1, TST0, TST1, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab)
    rhs_TST1= eye * build_rhs_tst1(TSN,TNS,TNG, TNR, TSS,TSR, TSG, TNT0, TNT1, TST0, TST1, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab)
    
    
    
    # slope at t+dt
    
    # TSNop_dt = TSNop + dt * rhs_TSNop
    # TNSop_dt = TNSop + dt * rhs_TNSop
    # TSSop_dt = TSSop + dt * rhs_TSSop
    
    TSN_dt = TSN + dt * rhs_TSN
    
    TNS_dt = TNS + dt * rhs_TNS
    TNG_dt = TNG + dt * rhs_TNG
    TNR_dt = TNR + dt * rhs_TNR
    
    TSS_dt = TSS + dt * rhs_TSS
    TSG_dt = TSG + dt * rhs_TSG
    TSR_dt = TSR + dt * rhs_TSR
    
    TNT0_dt = TNT0 + dt*rhs_TNT0
    TNT1_dt = TNT1 + dt*rhs_TNT1
    
    TST0_dt = TST0 + dt*rhs_TST0
    TST1_dt = TST1 + dt*rhs_TST1
    
    
    # rhs_TSNop2 = eye * build_rhs_tsnop(TSNop_dt,TNSop_dt,TSSop_dt,Hs,Hsuv,Vpqk,Vuv)
    # rhs_TNSop2 = eye * build_rhs_tnsop(TSNop_dt,TNSop_dt,TSSop_dt,omega,Vpqk,Vk)
    # rhs_TSSop2 = eye * build_rhs_tssop(TSNop_dt,TNSop_dt,TSSop_dt,Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u,Vuv)
    
    
    rhs_TSN2 = eye * build_rhs_tsn(TSN_dt,TNS_dt,TSS_dt,Hs,Hsuv,Vpqk,Vuv)
    
    
    rhs_TNS2 = eye * build_rhs_tns(TSN_dt,TNS_dt,TSS_dt,omega,Vpqk,Vk)
    rhs_TNG2 = eye * build_rhs_tng(TNG_dt,TSS_dt,omega,Vpqk)
    rhs_TNR2 = eye * build_rhs_tnr(TNR_dt,TSS_dt,omega,Vpqk)
    
    
    rhs_TSS2 = eye * build_rhs_tss(TSN_dt,TNS_dt,TSS_dt,TNG_dt,TNR_dt,TSG_dt,TSR_dt,Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u,Vuv)
    
    
    rhs_TSG2 = eye * build_rhs_tsg(TSN_dt,TNS_dt,TSS_dt,TNG_dt,TNR_dt,TSG_dt,TSR_dt,TNT0_dt, TNT1_dt, TST0_dt, TST1_dt,Hs,omega,Vpqk,Vu_v,Vu_u, Vu_ab)
    rhs_TSR2 = eye * build_rhs_tsr(TSN_dt,TNS_dt,TSS_dt,TNG_dt,TNR_dt,TSG_dt,TSR_dt,TNT0_dt, TNT1_dt, TST0_dt, TST1_dt,Hs,omega,Vpqk,Vu_v,Vu_u, Vu_ab)
    
    rhs_TNT02= eye * build_rhs_tnt0(TSN_dt,TNS_dt,TNG_dt, TNR_dt, TSS_dt,TSG_dt, TSR_dt, TNT0_dt, TNT1_dt, TST0_dt, TST1_dt, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab)
    rhs_TNT12= eye * build_rhs_tnt1(TSN_dt,TNS_dt,TNG_dt, TNR_dt, TSS_dt,TSG_dt, TSR_dt, TNT0_dt, TNT1_dt, TST0_dt, TST1_dt, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab)
    
    
    rhs_TST02= eye * build_rhs_tst0(TSN_dt,TNS_dt,TNG_dt, TNR_dt, TSS_dt,TSR_dt, TSG_dt, TNT0_dt, TNT1_dt, TST0_dt, TST1_dt, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab)
    rhs_TST12= eye * build_rhs_tst1(TSN_dt,TNS_dt,TNG_dt, TNR_dt, TSS_dt,TSR_dt, TSG_dt, TNT0_dt, TNT1_dt, TST0_dt, TST1_dt, Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u, Vu_ab)
    
    
    
    # update amplitudes
    time = time + dt
    # TSNop = TSNop + dt * (rhs_TSNop + rhs_TSNop2)/2.0
    # TNSop = TNSop + dt * (rhs_TNSop + rhs_TNSop2)/2.0
    # TSSop = TSSop + dt * (rhs_TSSop + rhs_TSSop2)/2.0
    
    
    TSN = TSN + dt * (rhs_TSN + rhs_TSN2)/2.0
    
    TNS = TNS + dt * (rhs_TNS + rhs_TNS2)/2.0
    TNG = TNG + dt * (rhs_TNG + rhs_TNG2)/2.0
    TNR = TNR + dt * (rhs_TNR + rhs_TNR2)/2.0
    
    TSS = TSS + dt * (rhs_TSS + rhs_TSS2)/2.0
    
    
    TSG = TSG + dt * (rhs_TSG + rhs_TSG2)/2.0
    TSR = TSR + dt * (rhs_TSR + rhs_TSR2)/2.0
    
    TNT0= TNT0 + dt*(rhs_TNT0+ rhs_TNT02)/2.0
    TNT1= TNT1 + dt*(rhs_TNT1+ rhs_TNT12)/2.0
    
    
    TST0= TST0 + dt*(rhs_TST0+ rhs_TST02)/2.0
    TST1= TST1 + dt*(rhs_TST1+ rhs_TST12)/2.0
    
    
    
    pt[epdmcc_iter-1]=TSN[0,0]-TSN[1,1]
    pto[epdmcc_iter-1]=TSNop[0,0]-TSNop[1,1]
    t[epdmcc_iter-1]=time
    
    
    
    
    # print results
    if (epdmcc_iter % 500 == 0):
        print('%d %3.5E %3.5E %3.5E %3.5E '\
            % (epdmcc_iter, (time*omegaR/(2.0*math.pi)), TSN[0,0].real+vf[0], TSN[1,1].real+vf[1], (TSN[0,0].real+vf[0]-TSN[1,1].real-vf[1])))
        #print(TNG-TNG.T)
        sys.stdout.flush()
print(t)

for i in range(len(t)):
    t[i]=t[i]*delta/np.pi/hbar


with open('3pmd40_temp_oc40_del_1_nogg.csv', 'a') as f:
    writer=csv.writer(f, delimiter=',')
    for i in range(len(t)):
        writer.writerow([t[i], pt[i]])

plt.plot(t,pt, 'b-.', label='three')
plt.plot(t,pto, 'r--', label ='one')
#plt.plot(time3, rho, 'k-', label='HEOM', alpha=.5)
plt.title('N=%3i, $\eta$=%.3f/$\Delta$' %(nvib, reorge/delta))
plt.xlim(0,10)
plt.ylim(-.5,1)
plt.legend()
plt.grid()
# end of code
