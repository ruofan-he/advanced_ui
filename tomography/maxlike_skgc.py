import numpy as np
from scipy.special import eval_hermite, factorial

hbar = 1

def proj_func(x, theta, max_photon=20):
    a = np.arange(max_photon+1)
    n,m = np.meshgrid(a, a)

    return np.sqrt(1/np.pi/hbar) * np.exp(-(x/np.sqrt(hbar))**2+1j*(n-m)*theta) * eval_hermite(n, x/np.sqrt(hbar)) * eval_hermite(m, x/np.sqrt(hbar)) /np.sqrt(factorial(n) * factorial(m) * np.power(2.0, n+m)) 

def maxlike(quad, phase, max_photon=20, max_iter=1000, conv_th = 0.001, dil_ep=np.inf):
    """
        Quantum state tomography with a maximum likelihood method by Lvovsky.
        Quadratures and phases must be arrayed in the same order.
        Iteration stops when the sum of square of absolute difference between two density matrices (rho_now, rho_next)
        is smaller than convergence threshold or the counter reaches maximum iteration number.
        Logarithmic likelihood value is printed before the result is returned.

        Parameters
        -------
        quad : 1d array-like
            measured quadrature values in 1d-array.
        phase : 1d array-like [rad]
            measured phases in 1d-array, with radian units.
        max_photon : int, optional
            maximum number of photon considered in Hilbert space.
        max_iter : int, optional
            maximum number of iteration.
        conv_th : float, optional
            threshold of convergence.
        dil_ep : float, optional
            Dilution ratio of likelihood iteration.
            By default, this value is set to be infinity and invalidate dilution.
            This value can take any positive number.
        -------

        Returns
        -------
        rho : 2d array-like
            density matrix which maximizes likelihood.
        -------
    """
    proj = np.stack([proj_func(q, theta, max_photon) for q,theta in zip(quad, phase)])
    print(proj.shape)
    print("projection operator prepared")

    I = np.eye(max_photon+1) / (max_photon+1)
    rho = np.eye(max_photon+1) / (max_photon+1)
    for i in range(max_iter):
        prob = np.tensordot(proj, rho.T, 2)
        R = np.sum(proj / prob[:,np.newaxis, np.newaxis], axis=0)
        R = R + np.conj(R.T)
        if dil_ep != np.inf and dil_ep > 0:
            R = (I + dil_ep * R) / (1 + dil_ep)
        R /= np.trace(R)
        
        rho_next = R.dot(rho.dot(R))
        rho_next = rho_next + np.conj(rho_next.T)
        rho_next /= np.trace(rho_next)

        err = np.sum(np.abs(rho - rho_next)**2)
        if err < conv_th:
            lik = np.sum(np.log(prob))
            print("Likelihood:%f"%lik.real)
            break
        print(i, err)
        rho = rho_next

    return rho_next

if __name__ == '__main__':
#    phase = np.load('gamma0p06_phase.npy')
    dir_name = 'data/'
    quad = np.load(dir_name+'gamma0p12_hd3_disp.npy')
    phase = np.load(dir_name+'gamma0p12_phase.npy')
#    import matplotlib.pyplot as plt
#    plt.hist(quad, bins=1000)
#    plt.show()
#    print(quad.shape)
#    phase = np.random.rand(len(quad)) * np.pi*2

    rho = maxlike(quad, phase, max_photon=20, conv_th=1e-8, max_iter=1000)
    np.save(dir_name+'gamma0p06_disp_rho.npy', rho)
