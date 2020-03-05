import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
import loader as ld
import spirals as sp
import plotting as pt
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
#Y0 = rng.standard_normal(size=(d, I))     #Placeholder. Y0 = initielle matrise med bilder
U0 = np.array((Wk, bk, w, mu))


#Med matrise som argument virker funksjonene på hvert element i matrisen
def eta(x): return 1/2 * (1 + np.tanh(x/2))
def d_eta(x): return 1/4 * (1 - (np.tanh(x/2))**2)
def sigma(x): return np.tanh(x)
def d_sigma(x): return 1 - (np.tanh(x))**2
def Z(x): return eta(np.transpose(x)@w + mu*one)        #x = siste Y_K


<<<<<<< HEAD
#############################################################################
            #Def av y0 og yk


def y0():
    #Lager en array av spirals, med gitt posisjon til true og false

    #Returns:
        #np.array -- arr[0] = [xpos, ypos] til false; arr[1] -- pos til True

    pos, bol = sp.get_data_spiral_2d()
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
    return np.array((np.array((xFalse, yFalse)), np.array((xTrue, yTure)))), C
=======
def big_j(Z, c):            #Fungerer for numpy arrays
    c = -1*c
    big_j = 0.5*la.norm(np.add(Z, c))**2
    return big_j
>>>>>>> 580d94d5aedb125959068fb0e7225764c41eca66

#Må returnere en tredimensjonal matrise, hvor den første dimensjonen svarer til iterasjon nr. k, og de to neste svarer til matrisen med bildet til det gitte laget k
def YK(Y0, K = K, sigma = sigma, h = h, Wk = Wk, bk = bk):
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

<<<<<<< HEAD


=======
>>>>>>> 580d94d5aedb125959068fb0e7225764c41eca66

#############################################################################
#Gradientberegninger

# Wk er tredimensjonell, bk er todimensjonell, Y er array med alle Y-matrisene
# Disse brukes for å regne ut gradienten for utregning av parametrene som brukes i neste lag
def gradient(Wk, bk, w, mu, Y):
    J_mu = d_eta(np.transpose(np.transpose(Y[-1])@w + mu*one)) * (Z(Y[-1]) - C)
<<<<<<< HEAD
    J_w = Y[-1]*((Z(Y[-1]) - C)*d_eta(np.transpose(Y[-1])@w + mu))
=======
    #print(w)
    #print(Y[-1].size)
    print("---")
    print(Y[-1])
    print((np.transpose(Y[-1])@w))
    print("---")
    print("w = ", w)
    J_w = Y[-1]@((Z(Y[-1]) - C)*d_eta(np.transpose(Y[-1])@w + mu))
    print("J_w = ", J_w)
    PK = np.array([np.outer(w, np.transpose((Z(Y[-1]) - C) * d_eta(np.transpose(Y[-1]) @ w + mu*one)))])
>>>>>>> 580d94d5aedb125959068fb0e7225764c41eca66

    PK = np.array([np.outer(w, np.transpose((Z(Y[-1]) - C) * d_eta(np.transpose(Y[-1]) @ w + mu*one)))])
    print(np.transpose((Z(Y[-1]) - C) * d_eta(np.transpose(Y[-1]) @ w + mu*one))
    for k in range(K-1, 0, -1):   #P0 brukes ikke så trenger ikke å regne den ut
        #Siden Pk regnes ut baklengs, stackes de baklengs inn i PK slik at alle Pk-ene stemmer overens med indekseringen i PK
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

<<<<<<< HEAD
=======
#J_mu, J_w, J_Wk, J_bk = gradient(Wk, bk, w, mu, YK(Y0))        #Y0 er placeholder
>>>>>>> 580d94d5aedb125959068fb0e7225764c41eca66


##############################################################################
#Adam decent algoritmen

<<<<<<< HEAD
=======
#kostfunksjon som måler hvor langt modellen er unna å klassifisere perfekt
#For I bilder

# big_j = 1/2 * np.sum(np.abs(Z-c)**2) == 1/2 la.norm(Z-c)**2
# big_j(U) s.4 pdf


def y0():
    """Lager en array av spirals, med gitt posisjon til true og false

    Returns:
        np.array -- arr[0] = [xpos, ypos] til false; arr[1] -- pos til True
    """
    pos, bol = sp.get_data_spiral_2d()
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
    print(X.size)
    print(X)
    Y = np.append(yFalse, yTure)
    print(Y.size)
    print(Y)
    Z = np.vstack((X, Y))
    print(Z.size)
    print(Z)
    return np.array((np.array((xFalse, yFalse)), np.array((xTrue, yTure)))), C, Z


>>>>>>> 580d94d5aedb125959068fb0e7225764c41eca66
#Definert ovenfor også, big_z er definert som Z(x), hvor x er input matrisen. Veldig sikker på at den fungerer korrekt, spurte studass om den
"""
def big_z(eta, mat_y, omega, mu, d):        #funk er ikke ferdig, noe mer må gjøres med mat_y
    big_z = eta(x)*(mat_y.transpose()*omega + mu*np.eye(d, k=0))    #numpy.eye(a, b) lager en axa matrise hvor k = b bestemmer subdiag/diag som blir 1 og resten 0. Nå er diagonalen 1.
    return big_z

# big_j fungerer
def big_j(big_z, c):                                # lager feil estimat for verdiene funnet fra Z.
    c = -1* c                                       # Bytter fortegn på np.array c.
    big_j = 0.5*la.norm(np.add(big_z, c))**2        # np.add summerer hvert i'te element i big_z og c sammen, tar deretter norm **2
    return big_j
"""


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


##############################################################################
                    #Selve programmet som kjenner igjen bildene

<<<<<<< HEAD
y0 = y0()
YK = y0[0]

=======
j = 0
tau = 0.01            #Læringsparameter. Vi skal bruke det som konvergerer raskest på intervallet [0.01,0.1]
#YK = Y0
>>>>>>> 580d94d5aedb125959068fb0e7225764c41eca66
# Tilfeldige startsverdier for vekter og bias står øverst i programmet


def u_j(U):
    #Regner ut U[j] hvor j går opp til N

    #Arguments:
        #N {int} -- # iterasjoner gjennom nettverket
        #U {np.array} -- U_0 -> start verdi for U = [Wk, bk, Ω, mu]

    #Returns:
        #U_j {np.array} -- U_j, hvor U = [Wk, bk, Ω, mu]

    tau = [.1, .01]

    U[0] = U[0] - tau[0]*U[0]
    U[1] = U[1] - tau[0]*U[1]
    U[2] = U[2] - tau[0]*U[2]
    U[3] = U[3] - tau[0]*U[3]
    return U

def algoritme(Y0,N,grad,K=K,sigma=sigma,h=h,Wk=Wk,bk=bk, w=w, mu = mu):
    j=0
    while j<N:
<<<<<<< HEAD
        Yk = YK(Y0,K=K,sigma=sigma,h=h,Wk=Wk,bk=bk)         # Array med K Yk matriser
        d_mu, d_omega, d_Wk, d_bk = grad(Wk,bk,omega,mu,YK) # Regner ut gradieinten for parametrene våre
        mu, omega, Wk, bk = u_j([d_mu,d_omega,d_Wk,d_bk])   # Kalkulerer ny verdi for parametrene
    return Yk, Wk, bk, omega, mu

algoritme(YK,2,gradient)
=======
        print("Tåpkuk")
        Yk = YK(Y0)         # Array med K Yk matriser
        d_mu, d_omega, d_Wk, d_bk = grad(Wk,bk,w,mu,Yk)                # Regner ut gradieinten for parametrene våre
        mu, w, Wk, bk = u_j([d_mu,d_omega,d_Wk,d_bk])
    return Yk, Wk, bk, w, mu

Y0 = y0()[2]
algoritme(Y0,5,gradient)
>>>>>>> 580d94d5aedb125959068fb0e7225764c41eca66
