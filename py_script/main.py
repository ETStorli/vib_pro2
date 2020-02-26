import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp


h = 0.1

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
    big_z = eta(x)*(mat_y.transpose()*omega + my*np.eye(d, 0))    #numpy.eye(a, b) lager en axa matrise hvor subdiagonal/diag b er 1 og resten 0. Nå er diagonalen 1.
    return big_z


# big_j fungerer
def big_j(big_z, c):                                # lager feil estimat for verdiene funnet fra Z.
    c = -1* c                                       # Bytter fortegn på np.array c.
    big_j = 0.5*la.norm(np.add(big_z, c))**2        # np.add summerer hvert i'te element i big_z og c sammen, tar deretter norm **2
    return big_j


#Forslag for adam algoritme. sorry toby, fjernet forslaget ditt da extension gjorde python triggered
#Har vet ikke om det er korrekt input

def adam_decent(gradient_J_U, U_j):
    beta_1 = 0.9
    beta_2 = 0.999
    alpha = 0.01
    litn_epsilon = 1E-8
    v_j = 0                                                 #Dette er v_0
    m_j = 0                                                 #dette er m_0
    # TODO: Finn ut ka vi iterer over
    for j in range(0,5):
        g_j = gradient_J_U                           # Gradienten er en matrise
        m_j = beta_1* m_j + (1-beta_1)* g_j          #antar m_j i likn nå er m_j fra forrige iterasjon
        v_j = beta_2*v_j +(1-beta_2)*(g_j*g_j)       #siste gj ledd er matrise mult.
        m_j_hatt = m_j/(1-beta_1**j)
        v_hatt = v_j/(1-beta_2**j)
        U_jp1 = U_j - alpha*(m_j_hatt/(np.sqrt(v_hatt)+litn_epsilon))       #U_jp1 = U_(j+1)
    return U_jp1




    ####### Ueder er en ikke ferdig puedo for hovedfunksjonen i lærings algoritmen
"""
def laer_tall(list_y0, K, tau, iterasjon lengde):
    generer mat_y0
    generer tilfeldig Wk, b_k, omega, my 
    
    while iterasjon mindre enn ønsket antall
        lag Y_k pluss 1 ved å summere mat_y0 og vekt, basis kritere
        
        lagre Y_k til neste iterasjon
        
        Beregn P_k fra likning 7 i hefte
        
        beregn "bitene av gradienten tilhørende projeksjon" i likn 5,6
        
        backtrace fra P_K og finn alle P_k
        
        beregn bidrag til gradientene fra liknin 9 og 10

        oppdater vektene og bias ved likn 4 eller adam metoden

    end

"""


























