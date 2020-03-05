import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
import random as rn
from numba import jit, njit, vectorize
from spirals import get_data_spiral_2d

print(get_data_spiral_2d(n_samples=200))
##############################################################################
                #Variabler som blir brukt gjennom prosjektet

d = 5
h = 0.1
<<<<<<< HEAD
#data =  #startsdataen vi har fått oppgitt
=======
d = 2
I = 10
k = np.array(I)
W_k = np.zeros(d*d).reshape([d,d])
>>>>>>> 3cef5398682a8a98d02722004e8f4cb3855fe317


<<<<<<< HEAD
##############################################################################
                    #Def av variabler for Y_0 og Y_k1
=======
#Y_(k+1) = Y_k = h*rho*(w_k*Y_k + b_k)
Y_0 = np.zeros(d*I).reshape(d,I)

def Y_k(d, I):
    pass
    

>>>>>>> 3cef5398682a8a98d02722004e8f4cb3855fe317

def eta(x):
    eta = 1/2 * (1 + np.tanh(x/2))
    return eta

def sigma(x):
    sigma = np.tanh(x)
    return sigma

def big_z(eta, mat_y, omega, my, d):        #funk er ikke ferdig, noe mer må gjøres med mat_y
    big_z = eta(x)*(mat_y.transpose()*omega + my*np.eye(d, k=0))    #numpy.eye(a, b) lager en axa matrise hvor subdiagonal/diag b er 1 og resten 0. Nå er diagonalen 1.
    return big_z

def big_j(big_z, c):            #Fungerer for numpy arrays
    c = -1*c
    big_j = 0.5*la.norm(np.add(big_z, c))**2
    return big_j



##############################################################################
                        #Def av Y_0 og Y_k1

def Make_Y_0(Y_0):
    pass


def Make_Yk(Y_matrix,sigma,h,Wk,bk):
    return Y_matrix+h*sigma*(Wk*Y_matrix+bk)


#############################################################################
                        #Gradientberegninger




##############################################################################
                        #Adam decent algoritmen

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
        #g[j] = np.gradient(big_j*U^{j})
        m[j] = beta_1*m[j-1] + (1-beta_1)*g[j]
        v[j] = beta_2*v[j-1] + (1-beta_2)*(g[j]*g[j])
        v_hat = v[j]/(1 - np.power(beta_2, j))
        #U^{j+1} = U^j - alpha* (m_hat[j]/(np.sqrt(v_hat[j]) + epsilon))
#Se også Stochastic gradient descent

#2.4) Gradient til big_j(U)
#∂J/∂W_k, ∂J/∂b_k, ∂J/∂w, ∂J/∂mu

##############################################################################
                    #Selve programmet som kjenner igjen bildene

Wk = np.eye(d,k=0)*rn.random()          #Def av d er øverst i programmet, linje 11
bk = np.ones(d)*rn.random()
omega = rn.random()
mu = rn.random()
