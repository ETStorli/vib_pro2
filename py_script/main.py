import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
from numba import jit, njit, vectorize

h = 0.1

#rho = np.tanh(x)

#Y_(k+1) = Y_k = h*rho*(w_k*Y_k + b_k)

def eta(x):
    eta = 1/2 * (1 + np.tanh(x/2))
    return eta



#kostfunksjon som måler hvor langt modellen er unna å klassifisere perfekt
#For I bilder
#//
# big_j = 1/2 * np.sum(np.abs(Z-c)**2) == 1/2 la.norm(Z-c)**2
# big_j(U) s.4 pdf
@njit
def adam_descent_alg():
    n = np.arange(10) #Vet ikke hva denne skal være enda
    beta_1, beta_2 = .9, .999
    alpha = .001
    epsilon = 1e-8
    v = np.empty_like(n)
    v_hat = np.empty_like(n)
    m = np.empty_like(n)
    m_hat = np.empty_like(n)
    g = np.empty_like(n)
    
    for j in range(1, n):
        pass
        #g[j] = np.gradient(big_j*U^{j})
        m[j] = beta_1*m[j-1] + (1-beta_1)*g[j]
        v[j] = beta_2*v[j-1] + (1-beta_2)*(g[j]*g[j]) 
        v_hat = v[j]/(1 - np.power(beta_2, j))
        #U^{j+1} = U^j - alpha* (m_hat[j]/(np.sqrt(v_hat[j]) + epsilon))
#Se også Stochastic gradient descent

#2.4) Gradient til big_j(U)
#∂J/∂W_k, ∂J/∂b_k, ∂J/∂w, ∂J/∂mu

