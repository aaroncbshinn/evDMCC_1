#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:45:30 2024

@author: acshinn
"""

import csv
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
import time
__authors__ = "Philip Shushkov"
__date__ = "2024-08-12"

np.set_printoptions(precision=8, linewidth=200, suppress=True)


# time3, rho = np.loadtxt(r'correctmed.csv', dtype=float,
#                         delimiter=',', usecols=(0, 1), unpack=True)
# time2, rho2 = np.loadtxt(r'new.csv', dtype=float,
#                           delimiter=',', usecols=(0, 1), unpack=True)

numpy_memory = 8

nstates = 4


# matrix elements of the spin Hamiltonian (cm^-1)
delta = 104.25
hbar = 5.034*1.05
Hs = np.zeros([nstates, nstates])

# temperature (cm-1)
temp = delta*2
coh = -1

c = .05*delta


# Rabi frequency [ps-1] = [THz]
omegaR = 2.0*delta/hbar

# electronic hamiltonian (HOLSTEIN MODEL)
for i in range(0, nstates-1):
    Hs[i, i+1] = coh
    Hs[i+1, i] = coh
    Hs[0, 0] = 0
    Hs[1, 1] = 0
    Hs[2, 2] = 0
    Hs[3, 3] = 0


print(Hs)
Hs = Hs*delta

gamma = 1*delta


# lead Voltage
Hpert = np.zeros([nstates, nstates])
Hpert[0, 0] = gamma
Hpert[1, 1] = 0#gamma
Hpert[2, 2] = 0#gamma
Hpert[3, 3] = gamma


# reorganization energy (cm-1) 2*\eta
reorge = delta/100

# bath characteristic frequency (cm-1)
omegac = delta*2.5
# number of bath oscillators
nvib = nstates
# maximum bath frequency
omegamax = omegac

beta = 1/temp

mod = int(nvib/nstates)
couplecut = 2

# Bath discretization from Wang, Song, Chandler, Miller
# QHO frequency [cm^-1]
omega = np.zeros(nvib)
for i in np.arange(omega.shape[0]):
    omega[i] = omegamax*((i % mod+1)/mod)**2

    # omega[i]=omegac*(i+1)*hbar

# Spin-vibrational coupling [cm^-1]^[1.5]
ck = np.zeros(mod)
for i in np.arange(ck.shape[0]):
    ck[i] = omega[i]*4.0*math.sqrt(omega[i]*omegamax)/(nvib*math.pi)
    ck[i] = ck[i]*0.5*reorge*(omega[i]*omegac)/(omega[i]**2+omegac**2)
    ck[i] = math.sqrt(ck[i])

print(ck)


# Spin-vibrational Hamiltonian [cm^-1]
Vpqk = np.zeros(nstates*nstates*nvib).reshape(nstates, nstates, nvib)
# print(Vpqk)
for i in range(0, nstates):
    for j in np.arange(mod*i, mod*i+mod):
        # Vpqk[0,0,i] = -ck[i]/math.sqrt(2.0*omega[i])
        # Vpqk[1,1,i] =  ck[i]/math.sqrt(2.0*omega[i])
        Vpqk[i, i, j] = c  # ck[j % mod]/math.sqrt(2.0*omega[j % mod])

print('Vpqk', Vpqk)
# chtim=np.arange(0, nvib, 1)
# plt.plot(chtim, ck)
# Bath discretization from Wang, Song, Chandler, Miller
# QHO frequency [cm^-1]


# Vibrational thermal factors
vb = np.zeros(nvib)
ub = np.zeros(nvib)
mu = np.array([-gamma,0,0,gamma])
for i in np.arange(0, nvib):
    vb[i] = 1.0/(math.exp(hbar*(omega[i % mod])/temp)-1.0)
    ub[i] = 1.0 + vb[i]
    
# print(vb)

# Electronic thermal factors
vf = np.zeros(nstates)
uf = np.zeros(nstates)
for i in np.arange(0, nstates,1):
    vf[i] = 1.0/(math.exp((Hs[i, i]-2*mu[i])/temp)+1.0)
    vf[0] = 0
    vf[3] = 1
    # vf[1] = .5
    # vf[2] = .5
    uf[i] = 1.0 - vf[i]


# Spin-vibrational Hamiltonian matrix elements with thermal factors
# Hs_ab*(v_b-v_a)
Hsuv = np.zeros(nstates*nstates).reshape(nstates, nstates)
for i in np.arange(Hsuv.shape[0]):
    for j in np.arange(Hsuv.shape[1]):
        Hsuv[i, j] = Hs[i, j]*(vf[j]-vf[i])


# Hpsuv = np.zeros(nstates*nstates).reshape(nstates,nstates)
# for i in np.arange(Hpsuv.shape[0]):
#     for j in np.arange(Hpsuv.shape[1]):
#         Hpsuv[i,j] = Hpert[i,j]*(vp[j]-vp[i])


# V_{ab,k}*(v_b-v_a)
Vuv = np.zeros(nstates*nstates*nvib).reshape(nstates, nstates, nvib)
for i in np.arange(Vuv.shape[0]):
    for j in np.arange(Vuv.shape[1]):
        for k in np.arange(Vuv.shape[2]):
            Vuv[i, j, k] = Vpqk[i, j, k]*(vf[j]-vf[i])

# V_{ii,k}*v_i
Vk = np.zeros(nvib)
for k in np.arange(Vpqk.shape[2]):
    for i in np.arange(Vpqk.shape[0]):
        Vk[k] += Vpqk[i, i, k]*vf[i]

# V_{ab,k}(u_k*u_a*v_b-v_k*u_b*v_a)
Vuvu = np.zeros(nstates*nstates*nvib).reshape(nstates, nstates, nvib)
for i in np.arange(Vuvu.shape[0]):
    for j in np.arange(Vuvu.shape[1]):
        for k in np.arange(Vuvu.shape[2]):
            Vuvu[i, j, k] = Vpqk[i, j, k]*(ub[k]*uf[i]*vf[j]-vb[k]*uf[j]*vf[i])

# V_{ab,k}*(u_a+v_k)
Vu_v = np.zeros(nstates*nstates*nvib).reshape(nstates, nstates, nvib)
for i in np.arange(Vu_v.shape[0]):
    for j in np.arange(Vu_v.shape[1]):
        for k in np.arange(Vu_v.shape[2]):
            Vu_v[i, j, k] = Vpqk[i, j, k]*(uf[i]+vb[k])

# V_{ab,k}*(u_k-u_a)
Vu_u = np.zeros(nstates*nstates*nvib).reshape(nstates, nstates, nvib)
for i in np.arange(Vu_u.shape[0]):
    for j in np.arange(Vu_u.shape[1]):
        for k in np.arange(Vu_u.shape[2]):
            Vu_u[i, j, k] = Vpqk[i, j, k]*(ub[k]-uf[i])


# Initialize amplitudes


# One Phonon Amplitudes
# TSNop = np.zeros(nstates*nstates).reshape(nstates,nstates) + 0.0j
# TSNop[0,0] = 1-uf[0] + 0.0j # (+1) state - population 1 donor
# TSNop[1,1] = 0-vf[1] + 0.0j # (-1) state - population 0 acceptor
# TSNop[2,2] = 0-vf[2] + 0.0j # (-1) state - population 0 acceptor
# TSNop[3,3] = 0-vf[3] + 0.0j # (-1) state - population 0 acceptor

TNSop = np.zeros(nvib) + 0.0j
# for i in np.arange(nvib):
#     TNSop[i] = -ck[i]/(math.sqrt(2.0*omega[i])*omega[i]) + 0.0j
# TSSop = np.zeros(nstates*nstates*nvib).reshape(nstates,nstates,nvib) + 0.0j


# Electronic amplitudes
# Initialize amplitudes
TSN = np.zeros(nstates*nstates).reshape(nstates, nstates) + 0.0j
TSN[0, 0] = .5-vf[0] + 0.0j  # (-1) state
TSN[1, 1] = .5-vf[1] + 0.0j  # (+1) state
TSN[2, 2] = .5-vf[2] + 0.0j  # (+1) state
TSN[3, 3] = .5-vf[3] + 0.0j  # (+1) state
# Vibrational amplitudes
# One-phonon
TNS = np.zeros(nvib) + 0.0j
# for i in np.arange(nvib):
#     TNS[i] = -ck[i]/(math.sqrt(2.0*omega[i])*omega[i]) + 0.0j

# Two-phonon Goepert
TNG = np.zeros(nvib*nvib).reshape(nvib, nvib) + 0.0j
# Two-phonon Raman
TNR = np.zeros(nvib*nvib).reshape(nvib, nvib) + 0.0j

# Electron-vibrational amplitudes
# Singles electron - one-phonon
TSS = np.zeros(nstates*nstates*nvib).reshape(nstates, nstates, nvib) + 0.0j
# Singles electron - two-phonon Goepert
TSG = np.zeros(nstates*nstates*nvib*nvib).reshape(nstates,
                                                  nstates, nvib, nvib) + 0.0j
# Singles electron - two-phonon Raman
TSR = np.zeros(nstates*nstates*nvib*nvib).reshape(nstates,
                                                  nstates, nvib, nvib) + 0.0j


# one phonon equations
def build_rhs_tsnop(TSNop, TNSop, TSSop, Hs, Hsuv, Vpqk, Vuv):
    TSSc = np.conj(TSSop)
    TNSs = TNSop + np.conj(TNSop)
    rhs_tsnop = np.einsum('ai,ib->ab', Hs, TSNop) - \
        np.einsum('ai,ib->ab', TSNop, Hs) + Hsuv
    rhs_tsnop += np.einsum('ail,ibl->ab', Vpqk, TSSop) + \
        np.einsum('ail,bil->ab', Vpqk, TSSc)
    rhs_tsnop -= np.einsum('bil,ail->ab', Vpqk, TSSop) + \
        np.einsum('bil,ial->ab', Vpqk, TSSc)
    rhs_tsnop += np.einsum('ail,ib,l->ab', Vpqk, TSNop, TNSs, optimize=True)
    rhs_tsnop -= np.einsum('bil,ai,l->ab', Vpqk, TSNop, TNSs, optimize=True)
    rhs_tsnop += np.einsum('abi,i->ab', Vuv, TNSs)
    return rhs_tsnop


def build_rhs_tnsop(TSNop, TNSop, TSSop, omega, Vpqk, Vk):
    rhs_tnsop = np.einsum('i,i->i', omega, TNSop) + \
        np.einsum('ijf,ij->f', Vpqk, TSNop) + Vk
    return rhs_tnsop


def build_rhs_tssop(TSNop, TNSop, TSSop, Hs, omega, Vuvu, Vpqk, Vu_v, Vu_u, Vuv):
    TNSs = TNSop + np.conj(TNSop)
    rhs_tssop = np.einsum('ai,ibf->abf', Hs, TSSop) - np.einsum('bi,aif->abf',
                                                                Hs, TSSop) + np.einsum('f,abf->abf', omega, TSSop)
    rhs_tssop += Vuvu
    rhs_tssop += np.einsum('aif,ib->abf', Vu_v, TSNop)
    rhs_tssop -= np.einsum('bif,ai->abf', Vu_u, TSNop)
    rhs_tssop -= np.einsum('ijf,aj,ib->abf', Vpqk, TSNop, TSNop)
    rhs_tssop += np.einsum('ail,l,ibf->abf', Vpqk, TNSs, TSSop, optimize=True)
    rhs_tssop -= np.einsum('ibl,l,aif->abf', Vpqk, TNSs, TSSop, optimize=True)
    return rhs_tssop


# Amplitude equations
def build_rhs_tsn(TSN, TNS, TSS, Hs, Hsuv, Vpqk, Vuv):
    TSSc = np.conj(TSS)
    TNSs = TNS + np.conj(TNS)
    rhs_tsn = np.einsum('ai,ib->ab', Hs, TSN) - \
        np.einsum('ai,ib->ab', TSN, Hs) + Hsuv
    rhs_tsn -= 1j*np.einsum('ai,ib->ab', Hpert, TSN)
    rhs_tsn -= 1j*np.einsum('bj,aj->ab', Hpert, TSN)
    rhs_tsn += np.einsum('ail,ibl->ab', Vpqk, TSS) + \
        np.einsum('ail,bil->ab', Vpqk, TSSc)
    rhs_tsn -= np.einsum('bil,ail->ab', Vpqk, TSS) + \
        np.einsum('bil,ial->ab', Vpqk, TSSc)
    rhs_tsn += np.einsum('ail,ib,l->ab', Vpqk, TSN, TNSs, optimize=True)
    rhs_tsn -= np.einsum('bil,ai,l->ab', Vpqk, TSN, TNSs, optimize=True)
    rhs_tsn += np.einsum('abi,i->ab', Vuv, TNSs)

    return rhs_tsn


def build_rhs_tns(TSN, TNS, TSS, omega, Vpqk, Vk):
    rhs_tns = np.einsum('i,i->i', omega, TNS) + \
        np.einsum('ijf,ij->f', Vpqk, TSN) + Vk
    return rhs_tns


def build_rhs_tng(TNG, TSS, omega, Vpqk):
    rhs_tng = np.einsum('i,ij->ij', omega, TNG) + \
        np.einsum('j,ij->ij', omega, TNG)
    rhs_tng += np.einsum('efd,efc->cd', Vpqk, TSS, optimize=True)
    rhs_tng += np.einsum('efc,efd->cd', Vpqk, TSS, optimize=True)
    return rhs_tng


def build_rhs_tnr(TNR, TSS, omega, Vpqk):
    TSSc = np.conj(TSS)
    rhs_tnr = np.einsum('i,ij->ij', omega, TNR) - \
        np.einsum('j,ij->ij', omega, TNR)
    rhs_tnr -= np.einsum('efd,efc->cd', Vpqk, TSS, optimize=True)
    rhs_tnr += np.einsum('efc,fed->cd', Vpqk, TSSc, optimize=True)
    return rhs_tnr


def build_rhs_tss(TSN, TNS, TSS, TNG, TNR, TSG, TSR, Hs, omega, Vuvu, Vpqk, Vu_v, Vu_u, Vuv):
    TNSs = TNS + np.conj(TNS)
    rhs_tss = np.einsum('ai,ibf->abf', Hs, TSS) - np.einsum('bi,aif->abf',
                                                            Hs, TSS) + np.einsum('f,abf->abf', omega, TSS)
    rhs_tss -= 1j*np.einsum('ai,ibf->abf', Hpert, TSS)
    rhs_tss -= 1j*np.einsum('bj,ajf->abf', Hpert, TSS)
    rhs_tss += Vuvu
    rhs_tss += np.einsum('aif,ib->abf', Vu_v, TSN)
    rhs_tss -= np.einsum('bif,ai->abf', Vu_u, TSN)
    rhs_tss -= np.einsum('ijf,aj,ib->abf', Vpqk, TSN, TSN)
    rhs_tss += np.einsum('ail,l,ibf->abf', Vpqk, TNSs, TSS, optimize=True)
    rhs_tss -= np.einsum('ibl,l,aif->abf', Vpqk, TNSs, TSS, optimize=True)
    rhs_tss += np.einsum('agk,gb,ck->abc', Vpqk, TSN, TNG, optimize=True)
    rhs_tss -= np.einsum('bhk,ah,ck->abc', Vpqk, TSN, TNG, optimize=True)
    rhs_tss += np.einsum('agk,gb,ck->abc', Vpqk, TSN, TNR, optimize=True)
    rhs_tss -= np.einsum('bhk,ah,ck->abc', Vpqk, TSN, TNR, optimize=True)
    rhs_tss += np.einsum('abk,ck->abc', Vuv, TNR, optimize=True)
    rhs_tss += np.einsum('abk,ck->abc', Vuv, TNG, optimize=True)
    rhs_tss += np.einsum('aek,ebck->abc', Vpqk, TSR, optimize=True)
    rhs_tss -= np.einsum('bfk,afck->abc', Vpqk, TSR, optimize=True)
    rhs_tss += np.einsum('aek,ebck->abc', Vpqk, TSG, optimize=True)
    rhs_tss -= np.einsum('bfk,afck->abc', Vpqk, TSG, optimize=True)
    return rhs_tss


def build_rhs_tsg(TSN, TNS, TSS, TNG, TNR, TSG, TSR, Hs, omega, Vpqk, Vu_v, Vu_u):
    Tns = np.conj(TNS)
    rhs_TSDG = np.einsum('ai,ibfe->abfe', Hs, TSG) - \
        np.einsum('bi,aife->abfe', Hs, TSG)
    rhs_TSDG += np.einsum('f,abfe->abfe', omega, TSG) + \
        np.einsum('e,abfe->abfe', omega, TSG)

    rhs_TSDG -= 1j*np.einsum('ai,ibef->abef', Hpert, TSG)
    rhs_TSDG -= 1j*np.einsum('bj,ajef->abef', Hpert, TSG)

    rhs_TSDG += np.einsum('aed,ebc->abcd', Vu_v, TSS, optimize=True)
    rhs_TSDG += np.einsum('aec,ebd->abcd', Vu_v, TSS, optimize=True)

    rhs_TSDG -= np.einsum('bfd,afc->abcd', Vu_u, TSS, optimize=True)
    rhs_TSDG -= np.einsum('bfc,afd->abcd', Vu_u, TSS, optimize=True)

    rhs_TSDG -= np.einsum('ihf,ibe,ah->abef', Vpqk, TSS, TSN, optimize=True)
    rhs_TSDG -= np.einsum('ihe,ibf,ah->abef', Vpqk, TSS, TSN, optimize=True)
    rhs_TSDG -= np.einsum('gjf,aje,gb->abef', Vpqk, TSS, TSN, optimize=True)
    rhs_TSDG -= np.einsum('gje,ajf,gb->abef', Vpqk, TSS, TSN, optimize=True)

    rhs_TSDG += np.einsum('aik,ibfe,k->abef', Vpqk, TSG, TNS, optimize=True)
    rhs_TSDG -= np.einsum('bjk,ajfe,k->abef', Vpqk, TSG, TNS, optimize=True)

    rhs_TSDG += np.einsum('aik,ibfe,k->abef', Vpqk, TSG, Tns, optimize=True)
    rhs_TSDG -= np.einsum('bjk,ajfe,k->abef', Vpqk, TSG, Tns, optimize=True)

    rhs_TSDG += np.einsum('aik,ibf,ek->abef', Vpqk, TSS, TNG, optimize=True)
    rhs_TSDG += np.einsum('aik,ibe,fk->abef', Vpqk, TSS, TNG, optimize=True)
    rhs_TSDG -= np.einsum('bjk,ajf,ek->abef', Vpqk, TSS, TNG, optimize=True)
    rhs_TSDG -= np.einsum('bjk,aje,fk->abef', Vpqk, TSS, TNG, optimize=True)

    rhs_TSDG += np.einsum('aik,ibf,ek->abef', Vpqk, TSS, TNR, optimize=True)
    rhs_TSDG += np.einsum('aik,ibe,fk->abef', Vpqk, TSS, TNR, optimize=True)
    rhs_TSDG -= np.einsum('bjk,ajf,ek->abef', Vpqk, TSS, TNR, optimize=True)
    rhs_TSDG -= np.einsum('bjk,aje,fk->abef', Vpqk, TSS, TNR, optimize=True)

    return rhs_TSDG


def build_rhs_tsr(TSN, TNS, TSS, TNG, TNR, TSG, TSR, Hs, omega, Vpqk, Vu_v, Vu_u):
    Tns = np.conj(TNS)
    Tng = np.conj(TNG)
    Tss = np.conj(TSS)
    rhs_TSDR = np.einsum('ai,ibfe->abfe', Hs, TSR) - \
        np.einsum('bi,aife->abfe', Hs, TSR)
    rhs_TSDR += np.einsum('f,abfe->abfe', omega, TSR) - \
        np.einsum('e,abfe->abfe', omega, TSR)

    rhs_TSDR -= 1j*np.einsum('ai,ibef->abef', Hpert, TSR)
    rhs_TSDR -= 1j*np.einsum('bj,ajef->abef', Hpert, TSR)

    rhs_TSDR += np.einsum('aif,ibe->abef', Vu_u, TSS, optimize=True)
    rhs_TSDR -= np.einsum('bjf,aje->abef', Vu_v, TSS, optimize=True)

    rhs_TSDR += np.einsum('aie,bif->abef', Vu_v, Tss, optimize=True)
    rhs_TSDR -= np.einsum('bje,jaf->abef', Vu_u, Tss, optimize=True)

    rhs_TSDR += np.einsum('ihf,ibe,ah->abef', Vpqk, TSS, TSN, optimize=True)
    rhs_TSDR += np.einsum('gjf,aje,gb->abef', Vpqk, TSS, TSN, optimize=True)
    rhs_TSDR -= np.einsum('ihe,bif,ah->abef', Vpqk, Tss, TSN, optimize=True)
    rhs_TSDR -= np.einsum('gje,jaf,gb->abef', Vpqk, Tss, TSN, optimize=True)

    rhs_TSDR += np.einsum('aik,ibef,k->abef', Vpqk, TSR, TNS, optimize=True)
    rhs_TSDR -= np.einsum('bjk,ajef,k->abef', Vpqk, TSR, TNS, optimize=True)

    rhs_TSDR += np.einsum('aik,ibef,k->abef', Vpqk, TSR, Tns, optimize=True)
    rhs_TSDR -= np.einsum('bjk,ajef,k->abef', Vpqk, TSR, Tns, optimize=True)

    rhs_TSDR += np.einsum('aik,ibe,kf->abef', Vpqk, TSS, TNR, optimize=True)
    rhs_TSDR -= np.einsum('bjk,aje,kf->abef', Vpqk, TSS, TNR, optimize=True)
    rhs_TSDR += np.einsum('aik,bif,ek->abef', Vpqk, Tss, TNR, optimize=True)
    rhs_TSDR -= np.einsum('bjk,jaf,ek->abef', Vpqk, Tss, TNR, optimize=True)

    rhs_TSDR += np.einsum('aik,bif,ek->abef', Vpqk, Tss, TNG, optimize=True)
    rhs_TSDR -= np.einsum('bjk,jaf,ek->abef', Vpqk, Tss, TNG, optimize=True)

    rhs_TSDR += np.einsum('aik,ibe,fk->abef', Vpqk, TSS, Tng, optimize=True)
    rhs_TSDR -= np.einsum('bjk,aje,fk->abef', Vpqk, TSS, Tng, optimize=True)

    return rhs_TSDR


# Iterations of time integration
print('\nStart Electron-Phonon DMCC iterations')
# Iterations of time integration
print('\nStart Electron-Phonon DMCC iterations')
# timestep (ps)
dt = 0.01*2.0*math.pi/omegaR
# number of iterations - 20 Rabi periods
maxiter = 2 * 1000
pt = np.zeros(maxiter)
pto = np.zeros(maxiter)
t = np.zeros(maxiter)
# time (ps)
time = 0.0

# imaginary unit (-i)
eye = 0.0 - 1.0j
# (-i)/\hbar
eye = eye/hbar
pt = np.zeros(maxiter)
p = np.zeros(maxiter)
pt1 = np.zeros(maxiter)
pt2 = np.zeros(maxiter)
pt3 = np.zeros(maxiter)
a1 = np.zeros(maxiter)
g1 = np.zeros(maxiter)
ptmf = np.zeros(maxiter)
pgmf = np.zeros(maxiter)
t = np.zeros(maxiter)
curr1 = np.zeros(maxiter)+1j


for epdmcc_iter in range(1, maxiter):
    # RK-2 integration method
    # slope at t
    time += dt
    # rhs_TSNop = eye * build_rhs_tsnop(TSNop,TNSop,TSSop,Hs,Hsuv,Vpqk,Vuv)
    # rhs_TNSop = eye * build_rhs_tnsop(TSNop,TNSop,TSSop,omega,Vpqk,Vk)
    # rhs_TSSop = eye * build_rhs_tssop(TSNop,TNSop,TSSop,Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u,Vuv)

    rhs_TSN = eye * build_rhs_tsn(TSN, TNS, TSS, Hs, Hsuv, Vpqk, Vuv)
    rhs_TNS = eye * build_rhs_tns(TSN, TNS, TSS, omega, Vpqk, Vk)
    rhs_TNG = eye * build_rhs_tng(TNG, TSS, omega, Vpqk)
    rhs_TNR = eye * build_rhs_tnr(TNR, TSS, omega, Vpqk)
    rhs_TSS = eye * build_rhs_tss(TSN, TNS, TSS, TNG, TNR,
                                  TSG, TSR, Hs, omega, Vuvu, Vpqk, Vu_v, Vu_u, Vuv)
    rhs_TSG = eye * build_rhs_tsg(TSN, TNS, TSS, TNG,
                                  TNR, TSG, TSR, Hs, omega, Vpqk, Vu_v, Vu_u)
    rhs_TSR = eye * build_rhs_tsr(TSN, TNS, TSS, TNG,
                                  TNR, TSG, TSR, Hs, omega, Vpqk, Vu_v, Vu_u)
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

    # rhs_TSNop2 = eye * build_rhs_tsnop(TSNop_dt,TNSop_dt,TSSop_dt,Hs,Hsuv,Vpqk,Vuv)
    # rhs_TNSop2 = eye * build_rhs_tnsop(TSNop_dt,TNSop_dt,TSSop_dt,omega,Vpqk,Vk)
    # rhs_TSSop2 = eye * build_rhs_tssop(TSNop_dt,TNSop_dt,TSSop_dt,Hs,omega,Vuvu,Vpqk,Vu_v,Vu_u,Vuv)

    rhs_TSN2 = eye * build_rhs_tsn(TSN_dt, TNS_dt, TSS_dt, Hs, Hsuv, Vpqk, Vuv)
    rhs_TNS2 = eye * build_rhs_tns(TSN_dt, TNS_dt, TSS_dt, omega, Vpqk, Vk)
    rhs_TNG2 = eye * build_rhs_tng(TNG_dt, TSS_dt, omega, Vpqk)
    rhs_TNR2 = eye * build_rhs_tnr(TNR_dt, TSS_dt, omega, Vpqk)
    rhs_TSS2 = eye * build_rhs_tss(TSN_dt, TNS_dt, TSS_dt, TNG_dt,
                                   TNR_dt, TSG_dt, TSR_dt, Hs, omega, Vuvu, Vpqk, Vu_v, Vu_u, Vuv)
    rhs_TSG2 = eye * build_rhs_tsg(TSN_dt, TNS_dt, TSS_dt, TNG_dt,
                                   TNR_dt, TSG_dt, TSR_dt, Hs, omega, Vpqk, Vu_v, Vu_u)
    rhs_TSR2 = eye * build_rhs_tsr(TSN_dt, TNS_dt, TSS_dt, TNG_dt,
                                   TNR_dt, TSG_dt, TSR_dt, Hs, omega, Vpqk, Vu_v, Vu_u)
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

    p[epdmcc_iter] = -(TSN[0, 0]+TSN[1, 1]+TSN[2, 2]+TSN[3, 3])
    pt[epdmcc_iter] = TSN[0, 0].real+vf[0]
    pt1[epdmcc_iter] = TSN[1, 1].real+vf[1]
    pt2[epdmcc_iter] = TSN[2, 2].real+vf[2]
    pt3[epdmcc_iter] = TSN[3, 3].real+vf[3]
    a1[epdmcc_iter] = np.average(np.abs(TNS))
    curr1[epdmcc_iter] = (TSN[1, 2].imag-TSN[2, 1].imag)
    t[epdmcc_iter] = time

    # print results
    if (epdmcc_iter % 1000 == 0):
        print('%d %3.5E %3.5E %3.5E %3.5E '
              % (epdmcc_iter, (time*omegaR/(2.0*math.pi)), TSN[0, 0].real+vf[0], TSN[1, 1].real+vf[1], (TSN[0, 0].real+vf[0]-TSN[1, 1].real-vf[1])))
        sys.stdout.flush()
print(p)



# temperature (cm-1) 
temp = 2*delta
# reorganization energy (cm-1)
reorge = delta*1
#number of spin states
nstates =4
# bath characteristic frequency (cm-1)
omegac = delta*2.5

# number of bath oscillators per site
nvib = nstates
#number of modes available to each oscillator
nosc =1


# maximum bath frequency 20-200 \omega_c
omegamax = 20*omegac
# reduced Planck constant (cm-1*ps)
#Hs = np.array([[delta,104.255],[104.255,-delta]])

coh=-1


delta = delta
hbar = 5.034*1.05


E = 0
T = temp


oc = omegac


bet = 1/T
gamma=gamma


#hbar=1
c =c
nparts=2

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
# vf = np.zeros(nstates)
# uf = np.zeros(nstates)
# #moving in to the grand canonical ensemble, we need the chemical potential
# mu=np.array([-gamma,0,0,gamma])
# for i in np.arange(0,nstates):
#     vf[i] = 1.0/(math.exp((Hs[i,i]-mu[i])/temp)+1.0)
#     uf[i] = 1.0 - vf[i]
#     vf[0]=0
#     vf[-1]=1
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
    

for i in range(len(t)):
    t[i]=t[i]*delta/hbar/np.pi

plt.plot(t1, Pt1, 'k--', label='site 1 exact')
plt.plot(t1, Pt2, 'r--', label='site 2 exact')
plt.plot(t1, Pt3, 'g--', label='site 3 exact')
plt.plot(t1, Pt4, 'b--', label='site 4 exact')

# plt.plot(t,p, 'c-', label='sum')

plt.plot(t, pt, 'k-', alpha=.7,  label='site 1')
plt.plot(t, pt1, 'r-', alpha=.7, label='site 2')
plt.plot(t, pt2, 'g-', alpha=.7,  label='site 3')
plt.plot(t, pt3, 'b-', alpha=.7, label='site 4')
#plt.legend(loc='center right', bbox_to_anchor=(1.25, .55),
#           ncol=1, fancybox=True, shadow=True)
plt.xlabel('time, ps')
plt.ylabel('Population of Sites')
plt.title('4-site Polaron Dynamics, c = %1.3f , \n diss diagonals(%1.2f, %1.2f,%1.2f,%1.2f), $\omega_c=%1.2f \Delta$, T=%2.3f$\Delta$ \n $v_1$=%.2f'
          % (Vpqk[0, 0, 0], Hpert[0, 0], Hpert[1, 1], Hpert[2, 2], Hpert[3, 3], omegac/delta, temp/delta, vf[0]))

# plt.ylim(ymin=0)
# plt.xlim(0,20)

# plt.legend()



plt.ylim(0, 1)

plt.xlim(0, 10)
plt.savefig('/Users/acshinn/Documents/polaronv01gp5.eps', format='eps')
plt.show()


# #plt.plot(t,p, 'y--', alpha=.6, label='Total')
plt.plot(t, curr1, 'b--')
plt.plot(t1,current, 'r-')
plt.axis('on')
# plt.plot(t1,Pt1, 'r--', alpha=.7, label='vibronic')
#plt.title('Current Density Operator')
plt.xlabel('Time ($t\Delta/\pi$)', fontname='Helvetica')
plt.ylabel('$\hat{J}_{23}(t)$', fontname='Helvetica', fontsize='15')
plt.xlim(0,10)
plt.ylim(-1,1)
plt.yticks(np.arange(-1,1.5,.5))
plt.xticks(np.arange(0,11,1))
# plt.hlines(0,0,20, colors='black', linewidth='0.8')

#plt.savefig('/Users/acshinn/Documents/currentpolaronv01g1.eps', format='eps')
# #plt.legend()
# # plt.title(r'Population Difference vs. Time, $\beta\Delta$= %3.3f, $\omega_c/\Delta$= %3.3f, c=%1.3f' %(delta/temp, omegac/delta*hbar, Vpqk[0,0,0]))

# plt.xlim(0,20)

# plt.show()


# plt.plot(t,pt, 'b-.', label='two')
# #plt.plot(t,pto, 'r--', label ='one')
# plt.plot(time3, rho, 'k-', label='HEOM', alpha=.5)
# plt.title('N=%3i, $\eta$=%.3f $\Delta$, $\omega_{max}$= %3i $\omega_c$ ' %(nvib, reorge/delta, omegamax/omegac))
# plt.xlim(0,20)
# plt.ylim(-1, 1)
# plt.legend()

# end of code
