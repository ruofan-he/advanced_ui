import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.special
from datetime import datetime
import json

class Wigner_fock:
    def __init__(self, n_max, q_range, q_res=0.01):
        self.n_max = n_max
        self.wavefunc_fock_instances = []
        self.q_range = q_range
        self.q_res = q_res
        N, M = np.meshgrid(np.arange(0, self.n_max+1, dtype=np.int), np.arange(0, self.n_max+1, dtype=np.int))
        self.N = N
        self.M = M
        self.I = np.diag(np.ones(self.n_max+1)) / (self.n_max+1)
        q = np.arange(-q_range, q_range, q_res)
        for n in range(n_max+1):
            wf = self.wavefunc_fock(n)
            self.wavefunc_fock_instances.append(sp.interpolate.interp1d(q, wf(q), kind='linear'))
        print("wavefunc instantiation end")

    def wavefunc_fock(self, n):
        return lambda x: sp.special.eval_hermite(n, x) / np.sqrt(2**n * sp.special.factorial(n) * np.sqrt(np.pi)) * np.exp(-x**2/2)

    def wigner_fock(self, m, n, X, Y):
        if m > n:
            m, n = n, m
            conj = True
        else:
            conj = False
        w = (-1) ** m * np.sqrt(2**(n - m) * sp.special.factorial(m) / sp.special.factorial(n)) * (X + 1.j * Y)**(n - m) * sp.special.genlaguerre(m, n - m)(2 * (X**2 + Y**2)) * np.exp(-(X**2 + Y**2)) / np.pi
        return np.conj(w) if conj else w

    def wigner_rho(self, rho, X, Y):
        w = np.zeros_like(X, dtype=np.complex128)
        for n in range(self.n_max+1):
            for m in range(self.n_max + 1):
                w += self.wigner_fock(m, n, X, Y) * rho[m, n]
        return w

    def maxlike_prepare(self, phases, quads):
        self.data_projectors = [ [ [] for n in range(self.n_max+1)] for m in range(self.n_max+1)]
        for phase, quad in zip(phases, quads):
            P = np.exp(1.j * (self.N - self.M) * phase)
            for n in range(self.n_max+1):
                for m in range(self.n_max+1):
                    self.data_projectors[n][m].append(P[n,m] * self.wavefunc_fock(n)(quad) * self.wavefunc_fock(m)(quad))
        for n in range(self.n_max+1):
            for m in range(self.n_max+1):
                self.data_projectors[n][m] = np.hstack(self.data_projectors[n][m])

        
    def maxlike_iter(self, rho):
        R = np.zeros((self.n_max+1, self.n_max+1), dtype=np.complex128)
        tr = np.zeros(len(self.data_projectors[0][0]), dtype=np.complex128)
        for n in range(self.n_max+1):
            for m in range(self.n_max+1):
                tr += self.data_projectors[n][m] * rho[n, m]
        for n in range(self.n_max+1):
            for m in range(self.n_max+1):
                R[n, m] = np.sum(self.data_projectors[n][m] / tr)
        R = R + np.conj(R.T)
        R /= np.trace(R)
        rho_next = R.dot(rho.dot(R))
        rho_next = rho_next + np.conj(rho_next.T)
        rho_next /= np.trace(np.real(rho_next))
        return rho_next, R

    def maxlike_clear(self):
        self.data_projectors = None

    def maxlike(self, phases, quads, max_iter=1000, th_conv=0.001):
        self.maxlike_prepare(phases, quads)
        print("maxlike: preparation end" + datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        #rho = np.diag(np.array([1, ] + [0, ]*self.n_max))
        rho = self.I
        res = []
        for i in range(max_iter):
            rho, R = self.maxlike_iter(rho)
            res.append(np.sum(np.abs(R - self.I)**2))
            if res[-1] < th_conv:
                break
                print("maxlike: converged. iter=" + str(i) + " " + datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        return rho, res