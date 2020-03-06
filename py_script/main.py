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
I = 200       #Antall bilder
h = 0.1         #Skrittlengde i transformasjonene
#C = np.ones(I)  #Vektor med skalarer på enten 1 eller 0 som forteller oss om "katt eller ikke katt"  Placeholder
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

        Z = np.array([np.append(xFalse, xTrue), np.append(yFalse, yTrue)])
    return np.array((np.array((xFalse, yFalse)), np.array((xTrue, yTrue)))), C, Z

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
def gradient(Wk, bk, w, mu, Y, C):
    J_mu = d_eta(np.transpose(np.transpose(Y[-1])@w + mu*one)) @ (Z(Y[-1]) - C)
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
    for k in range(1, K):
        b = np.array([bk[:, k]] * I).transpose()
        J_Wk = np.vstack((np.array([h*(PK[k] * d_sigma(Wk[k] @ Y[k] + b)) @ np.transpose(Y[k])]), J_Wk))
        J_bk = np.vstack((np.array([h * (PK[k] * d_sigma(Wk[k] @ Y[k] + b)) @ one]), J_bk))
    return J_Wk, J_bk.transpose(), J_w, J_mu

##############################################################################
def y0():
    """Lager en array av spirals, med gitt posisjon til true og false
    Returns:
        np.array -- arr[0] = [xpos, ypos] til false; arr[1] -- pos til True
    """
    pos, bol = sp.get_data_spiral_2d(I)
    posx, posy= pos[0], pos[1]

    xTrue = np.zeros(sum(bol))
    yTure = np.zeros(sum(bol))
    xFalse = np.zeros_like(xTrue)
    yFalse = np.zeros_like(xTrue)
    r = 0
    b_i = 0
    for idx, b in enumerate(bol):
        if b:
            xTrue[r] = posx[idx]
            yTure[r] = posy[idx]
            r += 1
        else:
            xFalse[b_i] = posx[idx]
            yFalse[b_i] = posy[idx]
            b_i += 1
        C = [False if x<len(xFalse) else True for x in range(len(xFalse) + len(xTrue))]
    X = np.append(xFalse, xTrue)
    Y = np.append(yFalse, yTure)
    Z = np.vstack((X, Y))
    return np.array((np.array((xFalse, yFalse)), np.array((xTrue, yTure)))), C, Z

############################################


def optimering(grad_U, U_j):           #Returnerer de oppdaterte parameterene for neste iterasjon
    tau = [.1, .01]
    """
    #print("U_J array :", U_j)
    print("Grad U[3]", grad_U[3])
    
    print("grad_U array: ", grad_U)
    """
    U_j[0] = U_j[0] - tau[0]*grad_U[0]
    U_j[1] = U_j[1] - tau[0]*grad_U[1]
    U_j[2] = U_j[2] - tau[0]*grad_U[2]
    U_j[3] = U_j[3] - tau[0]*grad_U[3]
    return U_j

##############################################################################
#Selve programmet som kjenner igjen bildene
# Tilfeldige startsverdier for vekter og bias står øverst i programmet


def algoritme(N,grad,K=K,sigma=sigma,h=h,Wk=Wk,bk=bk, w=w, mu = mu):
    j=0
    while j<N:
        _, C, Y0 = y0()
        Yk = YK(Y0)         # Array med K Yk matriser, kjører bildene igjennom alle lagene ved funk. YK
        d_Wk, d_bk, d_w, d_mu = grad(Wk,bk,w,mu,Yk,C)                # Regner ut gradieinten for parametrene våre
        Wk, bk, w, mu = optimering([d_Wk, d_bk, d_w, d_mu], [Wk, bk, w, mu])     # Oppdaterer parametrene vhp. u_j
        #print("wk, ", Wk,"bk: ", bk, "w: ", w, "my, ",  mu)
        j += 1
    return Yk[-1], Wk, bk, w, mu

Y_K, Wk, bk, w, mu = algoritme(5000, gradient)

# 1)
# Per nå kjøres SAMME Y0 igjennom modellen vår. Rett?
# Comment på commenten: Har gjort det slik at vi henter en ny Y0 for hver iterasjon

m, _, _= y0()
"""
plt.plot(m[0][0], m[0][1], '.')
plt.plot(m[1][0], m[1][1], '.')
plt.show()
"""

def split_YK(Y_k):
    x_false_true = np.split(Y_k[0], 2)
    y_false_true = np.split(Y_K[1], 2)
    Y_false = np.vstack((x_false_true[0], y_false_true[0]))
    Y_true = np.vstack((x_false_true[1], y_false_true[1]))

    return Y_false, Y_true


Y_false , Y_true = split_YK(Y_K)




plt.figure()
plt.plot(Y_false[0], Y_false[1], '.', color = "r")
plt.plot(Y_true[0], Y_true[1], '.', color = "b")
plt.show()
""""""

# 2)

# Må ha funk som kan plotte Y_K, det siste bildet etter alle lagene. Slik det er satt opp,
# er de første I/2 bildene False, og de siste I/2 bildene er True

# 3)
# Hva bruker vi big_J for? Er den testet og er den rett?
