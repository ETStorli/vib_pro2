import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp


h = 0.1

#rho = np.tanh(x)

#Y_(k+1) = Y_k = h*sigma*(w_k*Y_k + b_k)

def eta(x):
    eta = 1/2 * (1 + np.tanh(x/2))
    return eta


def sigma(x):
    sigma = np.tanh(x)
    return sigma

#kostfunksjon som måler hvor langt modellen er unna å klassifisere perfekt
#For I bilder
#//
# big_j = 1/2 * np.sum(np.abs(Z-c)**2) == 1/2 la.norm(Z-c)**2
# big_j(U) s.4 pdf


def big_z(eta, mat_y, omega, my, d):        #funk er ikke ferdig, noe mer må gjøres med mat_y
    big_z = eta*(mat_y.transpose()*omega + my*np.eye(d, 0))    #numpy.eye(a, b) lager en axa matrise hvor subdiagonal/diag b er 1 og resten 0. Nå er diagonalen 1.
    return big_z

def big_j(big_z, c):
    big_j = 0.5*la.norm(np.add(big_z, -1*c))**2
    return big_j



def mat_y_k1(y_0, wk, bk, sigma):
    y_k_p1 = y_0 + h*sigma()*(wk*y_0+bk)
    #Lagre matrise
    return y_k_p1










####### UNder er puedo for main i lærings algoritmen
"""
def laer_tall(list_y0, K, tau, iterasjon lengde):
    generer mat_y0
    generer tilfeldig Wk, b_k, omega, my 
    
    while iterasjon mindre enn ønsket antall
        lag Y_k pluss 1 ved å summere mat_y0 og vekt, basis kritere
        
        lagre Y_k til neste iterasjon
        
        Beregn P_k fra likning 7 i hefte
        
        beregn "bitene av gradienten tilhørende projeksjon" i likn 5,6
    


"""


























