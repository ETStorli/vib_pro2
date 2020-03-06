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
I = 200          #Antall bilder
h = 0.1         #Skrittlengde i transformasjonene
#C = np.ones(I)  #Vektor med skalarer på enten 1 eller 0 som forteller oss om "katt eller ikke katt"  Placeholder
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


def big_j(Z, c):            #Fungerer for numpy arrays
    c = -1*c
    big_j = 0.5*la.norm(np.add(Z, c))**2
    return big_j

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


#############################################################################
#Gradientberegninger

# Wk er tredimensjonell, bk er todimensjonell, Y er array med alle Y-matrisene
# Disse brukes for å regne ut gradienten for utregning av parametrene som brukes i neste lag
def gradient(Wk, bk, w, mu, Y, C):
    J_mu = d_eta(np.transpose(np.transpose(Y[-1])@w + mu*one)) * (Z(Y[-1]) - C)
    J_w = Y[-1]@((Z(Y[-1]) - C)*d_eta(np.transpose(Y[-1])@w + mu))
    PK = np.array([np.outer(w, np.transpose((Z(Y[-1]) - C) * d_eta(np.transpose(Y[-1]) @ w + mu*one)))])

    for k in range(K-1, 0, -1):   #P0 brukes ikke så trenger ikke å regne den ut
        #Siden Pk regnes ut baklengs, stackes de baklengs inn i PK slik at alle Pk-ene stemmer overens med indekseringen i PK
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


#Her er gradient _J_U listen med J derivert på div element. U_j inneholder [W_K, B_k, w, my]
def adam_decent(gradient_J_U, U_j, j):              #her må man ta inn j som teller iterasjonstallet
    beta_1 = 0.9
    beta_2 = 0.999
    alpha = 0.01
    litn_epsilon = 1E-8
    for u in range(0, len(U_j)):                            #Tanken er å bevege seg gjennom de 4 variablene i U_j
        v_j = 0                                 # Dette er v_0
        m_j = 0                                 #dette er m_0
        if u == 2 or u == 3:                    #befinner seg i W_k når u = 0, og Bk når u = 1
            g_j = gradient_J_U[u] ** j                          # Gradienten er en matrise
            m_j = beta_1 * m_j + (1 - beta_1) * g_j             # antar m_j i likn nå er m_j fra forrige iterasjon
            v_j = beta_2 * v_j + (1 - beta_2) * (g_j * g_j)     # siste gj ledd er matrise mult.
            m_j_hatt = m_j / (1 - beta_1 ** j)
            v_hatt = v_j / (1 - beta_2 ** j)
            for i in len(U_j[u]):                               #Itererer gjennom alle de k ulike matrisene eller bk verdiene
                U_j[u][i] = U_j[u][i] - alpha * (m_j_hatt / (np.sqrt (v_hatt) + litn_epsilon))  # U_jp1 = U_(j+1)
        elif u == 0 or u == 1:                                  #Her er det tanken å iterere gjennom alle W_k i W_K men at len = 1 for alle andre verdier i U_j
            g_j = gradient_J_U[u]**j                            # Gradienten er en matrise
            m_j = beta_1* m_j + (1-beta_1)* g_j                 #antar m_j i likn nå er m_j fra forrige iterasjon
            v_j = beta_2*v_j +(1-beta_2)*(g_j*g_j)              #siste gj ledd er matrise mult.
            m_j_hatt = m_j/(1-beta_1**j)
            v_hatt = v_j/(1-beta_2**j)
            U_j[u] = U_j[u] - alpha*(m_j_hatt/(np.sqrt(v_hatt)+litn_epsilon))       #U_jp1 = U_(j+1)
    return U_j

j = 0
tau = 0.01            #Læringsparameter. Vi skal bruke det som konvergerer raskest på intervallet [0.01,0.1]

def u_j(U):           #Returnerer de oppdaterte parameterene for neste iterasjon
    tau = [.1, .01]

    U[0] = U[0] - tau[0]*U[0]
    U[1] = U[1] - tau[0]*U[1]
    U[2] = U[2] - tau[0]*U[2]
    U[3] = U[3] - tau[0]*U[3]
    return U

##############################################################################
#Selve programmet som kjenner igjen bildene
# Tilfeldige startsverdier for vekter og bias står øverst i programmet


def algoritme(N,grad,K=K,sigma=sigma,h=h,Wk=Wk,bk=bk, w=w, mu = mu):
    j=0
    while j<N:
        _, C, Y0 = y0()
        Yk = YK(Y0)         # Array med K Yk matriser, kjører bildene igjennom alle lagene ved funk. YK
        d_Wk, d_bk, d_w, d_mu = grad(Wk,bk,w,mu,Yk,C)                # Regner ut gradieinten for parametrene våre
        Wk, bk, w, mu = u_j([d_Wk, d_bk, d_w, d_mu])                 # Oppdaterer parametrene vhp. u_j
        j += 1
    return Yk[-1], Wk, bk, w, mu

Y_K, Wk, bk, w, mu = algoritme(5,gradient)

# 1)
# Per nå kjøres SAMME Y0 igjennom modellen vår. Rett?
# Comment på commenten: Har gjort det slik at vi henter en ny Y0 for hver iterasjon

m, _, _= y0()
plt.plot(m[0][0], m[0][1], '.')
plt.plot(m[1][0], m[1][1], '.')
plt.show()

# 2)
# Må ha funk som kan plotte Y_K, det siste bildet etter alle lagene. Slik det er satt opp,
# er de første I/2 bildene False, og de siste I/2 bildene er True

# 3)
# Hva bruker vi big_J for? Er den testet og er den rett?
