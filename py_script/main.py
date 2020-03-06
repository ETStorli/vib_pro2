import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
#import loader as ld
import spirals as sp
#import plotting as pt
import random as rn
rng = np.random.default_rng()

K = 3           #Antall lag
d = 2           #Antall piksel-elementer til hvert bilde. Hvert bilde er stablet opp i en vektor av lengde d
I = 10          #Antall bilder
h = 0.1         #Skrittlengde i transformasjonene
C = np.ones(I)  #Vektor med skalarer på enten 1 eller 0 som forteller oss om "katt eller ikke katt"
Wk = rng.standard_normal(size=(K, d, d))
w = rng.standard_normal(size=(d))
mu = rng.standard_normal(size=1)
one = np.ones(I)
bk = rng.standard_normal(size=(d, K))
U0 = np.array((Wk, bk, w, mu))

#Med matrise som argument virker funksjonene på hvert element i matrisen
def eta(x): return 1/2 * (1 + np.tanh(x/2))
def d_eta(x): return 1/4 * (1 - (np.tanh(x/2))**2)
def sigma(x): return np.tanh(x)
def d_sigma(x): return 1 - (np.tanh(x))**2
def Z(x): return eta(np.transpose(x)@w + mu*one)        #x = siste Y_K

def y0():
    """Lager en array av spirals, med gitt posisjon til true og false

    Returns:
        np.array -- arr[0] = [xpos, ypos] til false; arr[1] -- pos til True
    """
    pos, bol = sp.get_data_spiral_2d(I)
    posx, posy = pos[0], pos[1]

    xTrue = np.zeros(sum(bol))
    yTrue = np.zeros(sum(bol))
    xFalse = np.zeros_like(xTrue)
    yFalse = np.zeros_like(xTrue)
    r = 0
    b_i = 0
    for idx, b in enumerate(bol):
        if b:
            xTrue[r] = posx[idx]
            yTrue[r] = posy[idx]
            r += 1
        else:
            xFalse[b_i] = posx[idx]
            yFalse[b_i] = posy[idx]
            b_i += 1
        C = [False if x < len(xFalse) else True for x in range(
            len(xFalse) + len(xTrue))]

        Y = np.array([np.append(xFalse, xTrue), np.append(yFalse, yTrue)])
    return np.array((np.array((xFalse, yFalse)), np.array((xTrue, yTrue)))), C, Y

#Må returnere en tredimensjonal matrise, hvor den første dimensjonen svarer til iterasjon nr. k, og de to neste svarer til matrisen med bildet til det gitte laget k

y_start = y0()[2]

def YK(Y0=y_start, K=K, sigma=sigma, h=h, Wk=Wk, bk=bk):
    Y_out = np.random.rand(K, d, I)
    Y = Y0
    for k in range(K):
        X = Wk[k] @ Y
        # bk er en kolonnevektor fra b, men leses som radvektor etter at vi har hentet den ut. Derfor transponerer vi
        # Vi ganger med I for å få en matrise, som etter å ha transponert, blir en matrise med I bk-kolonnevektorer. Må gjøre det slik for at adderingen skal funke
        X = X + np.array([bk[:, k]] * I).transpose()      #bk[:, k] leses: alle rader, i kolonne k. Henter altså ut kolonnevektor k fra matrisen bk
        Y_out[k] = Y + h*sigma(X)
        Y = Y_out[k]
    return Y_out


# Wk er tredimensjonell, bk er todimensjonell, Y er array med alle Y-matrisene
# Disse brukes for å regne ut gradienten for utregning av parametrene som brukes i neste lag
def gradient(Y, Wk=Wk, bk=bk, w=w, mu=mu):
    J_mu = d_eta(np.transpose(np.transpose(Y[-1])@w + mu*one)) * (Z(Y[-1]) - C)
    J_w = Y[-1]@((Z(Y[-1]) - C)*d_eta(np.transpose(Y[-1])@w + mu))

    PK = np.array([np.outer(w, np.transpose((Z(Y[-1]) - C) * d_eta(np.transpose(Y[-1]) @ w + mu*one)))])

    for k in range(K-1, 0, -1):   #P0 brukes ikke så trenger ikke å regne den ut
        #Siden Pk regnes ut baklengs, stackes de baklengs inn i PK slik at alle Pk-ene stemmer overens med indekseringen i PK
        #! Andre iter blir file hos viktor
        b = np.array([bk[:, k]] * I).transpose()
        PK = np.vstack((np.array([PK[0] + h*np.transpose(Wk[k])@(d_sigma(Wk[k] @ Y[k] + b) * PK[0])]), PK))

    b = np.array([bk[:, 0]] * I).transpose()
    J_Wk = np.array([h*(PK[0] * d_sigma(Wk[0] @ Y[0] + b)) @ np.transpose(Y[0])])
    J_bk = np.array([h*(PK[0] * d_sigma(Wk[0] @ Y[0] + b)) @ one])
    for i in range(1, K):
        b = np.array([bk[:, k]] * I).transpose()
        J_Wk = np.vstack((np.array([h*(PK[k] * d_sigma(Wk[k] @ Y[k] + b)) @ np.transpose(Y[k])]), J_Wk))
        J_bk = np.vstack((np.array([h * (PK[k] * d_sigma(Wk[k] @ Y[k] + b)) @ one]), J_bk))
    return J_mu, J_w, J_Wk, J_bk

#J_mu, J_w, J_Wk, J_bk = gradient(Wk, bk, w, mu, YK(Y0))        #Y0 er placeholder

y_k = YK()
def u_j(U):
    """Regner ut U[j] hvor j går opp til N

    Arguments:
        N {int} -- # iterasjoner gjennom nettverket
        U {np.array} -- U_0 -> start verdi for U = [Wk, bk, Ω, mu]

    Returns:
        U_j {np.array} -- U_j, hvor U = [Wk, bk, Ω, mu]
    """
    tau = [.1, .01]

    U[0] = U[0] - tau[0]*gradient(y_k, U[0])
    U[1] = U[1] - tau[0]*gradient(y_k, U[1])
    U[2] = U[2] - tau[0]*gradient(y_k, U[2])
    U[3] = U[3] - tau[0]*gradient(y_k, U[3])
    return U


def algoritme(y_0,N,grad,K=K,sigma=sigma,h=h,Wk=Wk,bk=bk, w=w, mu=mu):
    j=0
    while j<N:
        Yk = YK(y_0,K=K,sigma=sigma,h=h,Wk=Wk,bk=bk)         # Array med K Yk matriser
        d_mu, d_omega, d_Wk, d_bk = grad(Wk, bk, w, mu, YK)                # Regner ut gradieinten for parametrene våre
        mu, w, Wk, bk = u_j([d_mu, d_omega, d_Wk, d_bk])
    return Yk, Wk, bk, w, mu




