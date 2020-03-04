import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp


h = 0.1

def eta(x):
    eta = 1/2 * (1 + np.tanh(x/2))
    return eta

## Sigma er impementert i make Y_k+1

def make_y_kp1(mat_y0, wk, bk):
    mat_y_kp1 = mat_y0 + h * np.tanh(wk*mat_y0 + bk)
    return mat_y_kp1


##
def big_z(eta, mat_y, omega, my, big_i):        #funk er ikke ferdig, noe mer må gjøres med mat_y
    big_z = eta((mat_y.transpose()*omega + my*np.ones(1, big_i)))    #numpy.eye(a, b) lager en axa matrise hvor k = b bestemmer subdiag/diag som blir 1 og resten 0. Nå er diagonalen 1.
    return big_z


#kostfunksjon som måler hvor langt modellen er unna å klassifisere perfekt
#For I bilder
def big_j(big_z, c):                                # lager feil estimat for verdiene funnet fra Z.
    c = -1 * c                                       # Bytter fortegn på np.array c.
    big_j = 0.5*la.norm(np.add(big_z, c))**2        # np.add summerer hvert i'te element i big_z og c sammen, tar deretter norm **2
    return big_j

def gradient_u():




## den simple funksjonen for optimering
def optimise_u(U, tau):
    u_jp1= U- tau*big_j(U)
    return u_jp1



#Forslag for adam algoritme. sorry toby, fjernet forslaget ditt da extension gjorde python triggered
#Har vet ikke om det er korrekt input


#Her er gradient _J_U listen med J derivert på div element i
def adam_decent(gradient_J_U, U_j):
    beta_1 = 0.9
    beta_2 = 0.999
    alpha = 0.01
    litn_epsilon = 1E-8
    v_j = 0                                                 #Dette er v_0
    m_j = 0                                                 #dette er m_0
    # TODO: Finn ut ka vi iterer over
    for u in range(0, len(U_j)):                            #Tanken er å bevege seg gjennom de 4 variablene i U_j
        for j in range(len(U_j)):                        #Her er det tanken å iterere gjennom alle W_k i W_K men at len = 1 for alle andre verdier i U_j
            g_j = gradient_J_U[u]**j                           # Gradienten er en matrise
            m_j = beta_1* m_j + (1-beta_1)* g_j          #antar m_j i likn nå er m_j fra forrige iterasjon
            v_j = beta_2*v_j +(1-beta_2)*(g_j*g_j)       #siste gj ledd er matrise mult.
            m_j_hatt = m_j/(1-beta_1**j)
            v_hatt = v_j/(1-beta_2**j)
            U_j[u][j] = U_j[u][j] - alpha*(m_j_hatt/(np.sqrt(v_hatt)+litn_epsilon))       #U_jp1 = U_(j+1)
    return U_j





##Placeholder funkcjoner
def make_y0(v):
    a = 0
    return 0

def gen_rand_konst():
    return 1,2,3,4

def calc_big_p_k():
    return 1

def main(list_y0, K, tau, iterasjon_lim):
    mat_yK = make_y0(list_y0)
    mat_list_wk, vec_list_bk, omega, my = gen_rand_konst()            # mat_list wk er en array med K, dxd matriser

    #vec_list_bk er en array med 15 dx1 vektorer bk

    list_yk = []   #
    i = 0
    while i != 1:
        for i in range(0, K):
           mat_yK = make_y_kp1(mat_yK, mat_list_wk[i], vec_list_bk[i])
           list_yk.append(mat_yK)
        big_p_k = calc_big_p_k()
    return 1




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


























