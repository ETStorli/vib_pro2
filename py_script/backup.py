import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
#import loader as ld
import spirals as sp
#import plotting as pt
import random as rn
rng = np.random.randn
from plotting import plot_progression

K = 15          #Antall lag
d = 2           #Antall piksel-elementer til hvert bilde. Hvert bilde er stablet opp i en vektor av lengde d
I = 200         #Antall bilder
h = 0.1         #Skrittlengde i transformasjonene
Wk = rng(K, d, d)
w = rng(d)
mu = rng(1)
one = np.ones(I)
bk = rng(d, K)
U0 = np.array((Wk, bk, w, mu))
y0, C = sp.get_data_spiral_2d(I)
C = np.reshape(C, I)

#Med matrise som argument virker funksjonene på hvert element i matrisen
def eta(x): return 1/2 * (1 + np.tanh(x/2))
def d_eta(x): return 1/4 * (1 - (np.tanh(x/2))**2)
def sigma(x): return np.tanh(x)
def d_sigma(x): return 1 - (np.tanh(x))**2
def Z(x): return eta(np.transpose(x)@w + mu*one)        #x = siste Y_K


def YK(y0, Wk, bk, sigma=sigma, h=h, K=K):
    Y_out = np.random.rand(K, d, I)
    Y = y0
    Y_out[0] = y0
    for k in range(1,K):
        X = Wk[k] @ Y
        # bk er en kolonnevektor fra b, men leses som radvektor etter at vi har hentet den ut. Derfor transponerer vi
        # Vi ganger med I for å få en matrise, som etter å ha transponert, blir en matrise med I bk-kolonnevektorer. Må gjøre det slik for at adderingen skal funke
        '''
        Har gjort om på b, før var det dette:
        b = np.array([bk[:, k]] * I).transpose()
        Jeg vet ikke hvorfor det var sånn
        '''
        X = X + np.array([bk[:, k]]).transpose()      #bk[:, k] leses: alle rader, i kolonne k. Henter altså ut kolonnevektor k fra matrisen bk
        Y_out[k] = Y + h*sigma(X)
        Y = Y_out[k]
    return Y_out

def gradient(Wk, bk, w, mu, Y, C):
    J_mu = np.transpose(d_eta(np.transpose(Y[-1])@w + mu*one)) @ (Z(Y[-1]) - C)
    J_w = Y[-1]@((Z(Y[-1]) - C)*d_eta(np.transpose(Y[-1])@w + mu))
    PK = np.array([np.outer(w, np.transpose((Z(Y[-1]) - C) * d_eta(np.transpose(Y[-1]) @ w + mu*one)))])

    #print(np.transpose((Z(Y[-1]) - C) * d_eta(np.transpose(Y[-1]) @ w + mu*one)))
    for k in range(K-1, 0, -1):   #P0 brukes ikke så trenger ikke å regne den ut
        #Siden Pk regnes ut baklengs, stackes de baklengs inn i PK slik at alle Pk-ene stemmer overens med indekseringen i PK
        #! Andre iter blir file hos viktor
        '''
        Har gjort om på b, før var det dette:
        b = np.array([bk[:, k]] * I).transpose()
        Jeg vet ikke hvorfor det var sånn
        '''
        b = np.array([bk[:, k]]).transpose()
        PK = np.vstack((np.array([PK[0] + h*np.transpose(Wk[k])@(d_sigma(Wk[k] @ Y[k] + b) * PK[0])]), PK))

    #FIXME: Her er det en grov feil. Viktor sier vi ikke er iteressert i P0 (som er sant nok),
    #       men dette er faktisk P1 ettersom han har ekskludert Y0 tidligere

    '''
    Har gjort om på b, før var det dette:
    b = np.array([bk[:, k]] * I).transpose()
    Jeg vet ikke hvorfor det var sånn
    '''
    b = np.array([bk[:, 0]]).transpose()
    J_Wk = np.array([h*(PK[0] * d_sigma(Wk[0] @ Y[0] + b)) @ np.transpose(Y[0])])
    J_bk = np.array([h*(PK[0] * d_sigma(Wk[0] @ Y[0] + b)) @ one])

    for k in range(1, K):
        '''
        Har gjort om på b, før var det dette:
        b = np.array([bk[:, k]] * I).transpose()
        Jeg vet ikke hvorfor det var sånn
        '''
        b = np.array([bk[:, k]]).transpose()
        J_Wk = np.vstack((J_Wk, np.array([h*(PK[k] * d_sigma(Wk[k-1] @ Y[k-1] + b)) @ np.transpose(Y[k-1])])))
        J_bk = np.vstack((J_bk, (np.array([h * (PK[k] * d_sigma(Wk[k-1] @ Y[k-1] + b)) @ one]))))
        #J_mu, J_w, J_Wk, J_bk = gradient(Wk, bk, w, mu, YK(Y0))        #Y0 er placeholder
    return J_Wk, J_bk.transpose(), J_w, J_mu

def optimering(grad_U, U_j):           #Returnerer de oppdaterte parameterene for neste iterasjon
    tau = 0.04
    """
    #print("U_J array :", U_j)
    print("Grad U[3]", grad_U[3])

    print("grad_U array: ", grad_U)
    """
    for i in range(K-1):
        #print("Wk før:",U_j[0])
        U_j[0][i] = U_j[0][i] - tau*grad_U[0][i]      #Wk
        #print("Wk etter:",U_j[0])
        #print("bk før:",U_j[1])
        U_j[1][:,i] = U_j[1][:,i] - tau*grad_U[1][:,i]      #bk
        #print("bk etter:",U_j[1])
    #print("w før:", U_j[2])
    U_j[2] = U_j[2] - tau*grad_U[2]      #w
    #print("w etter:", U_j[2])
    #print("mu før:",U_j[3])
    U_j[3] = U_j[3] - tau*grad_U[3]      #mu
    #'print("mu etter:",U_j[3])

    return U_j



def algoritme(N,grad, y0=y0, C=C, K=K,sigma=sigma,h=h,Wk=Wk,bk=bk, w=w, mu=mu):
    j=0
    C, Y0 = C, y0
    while j<N:
        Yk = YK(Y0,Wk,bk)         # Array med K Yk matriser, kjører bildene igjennom alle lagene ved funk. YK
        d_Wk, d_bk, d_w, d_mu = grad(Wk,bk,w,mu,Yk,C)                # Regner ut gradieinten for parametrene våre
        Wk, bk, w, mu = optimering([d_Wk, d_bk, d_w, d_mu], [Wk, bk, w, mu])     # Oppdaterer parametrene vhp. u_j
        #print("wk, ", Wk,"bk: ", bk, "w: ", w, "my, ",  mu)
        j += 1
    return Yk, Wk, bk, w, mu

def split_YK(Y_k):
    x_false_true = np.split(Y_k[0], 2)
    y_false_true = np.split(Y_K[1], 2)
    Y_false = np.vstack((x_false_true[0], y_false_true[0]))
    Y_true = np.vstack((x_false_true[1], y_false_true[1]))

    return Y_false, Y_true



# 1)
# Per nå kjøres SAMME Y0 igjennom modellen vår. Rett?
# Comment på commenten: Har gjort det slik at vi henter en ny Y0 for hver iterasjon

# 2)

#print(algoritme(2,gradient))

plot_progression(algoritme(1000,gradient)[0],C)
# Må ha funk som kan plotte Y_K, det siste bildet etter alle lagene. Slik det er satt opp,
# er de første I/2 bildene False, og de siste I/2 bildene er True

# 3)
# Hva bruker vi big_J for? Er den testet og er den rett?
