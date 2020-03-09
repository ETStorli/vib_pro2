import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
import loader as ld
import spirals as sp
from plotting import plot_progression, plot_model, plot_separation
import random as rn

rng = np.random.randn

K = 15  # Antall lag
d = 2  # Antall piksel-elementer til hvert bilde. Hvert bilde er stablet opp i en vektor av lengde d
I = 200  # Antall bilder
h = 0.1  # Skrittlengde i transformasjonene
Wk = rng(K, d, d)
w = rng(d)
mu = rng(1)
one = np.ones(I)
bk = rng(d, K)
U0 = np.array ((Wk, bk, w, mu))
y0, C1 = sp.get_data_spiral_2d(I)
C = np.reshape(C1, I)

# Med matrise som argument virker funksjonene på hvert element i matrisen
def eta(x): return 1 / 2 * (1 + np.tanh(x / 2))


def d_eta(x): return (1-(np.tanh(x/2))**2)/4


def sigma(x): return np.tanh(x)


def d_sigma(x): return 1 - (np.tanh(x))**2
def Z(x, omega, mu): return eta(np.transpose(x) @ omega + mu * one)  # x = siste Y_K



def big_j(Z, c):            #Fungerer for numpy arrays
    c = -1*c
    big_j = 0.5*la.norm(np.add(Z, c))**2
    return big_j

#Må returnere en tredimensjonal matrise, hvor den første dimensjonen svarer til iterasjon nr. k, og de to neste svarer til matrisen med bildet til det gitte laget k
def YK(y0, K = K, sigma = sigma, h = h, Wk = Wk, bk = bk):
    Y_out = np.random.rand(K, d, I)
    Y = y0
    Y_out[0] = y0
    for k in range (1, K):
        # bk er en kolonnevektor fra b, men leses som radvektor etter at vi har hentet den ut. Derfor transponerer vi
        # Vi ganger med I for å få en matrise, som etter å ha transponert, blir en matrise med I bk-kolonnevektorer.
        # Må gjøre det slik for at adderingen skal funke
        # bk[:, k] leses: alle rader, i kolonne k. Henter altså ut kolonnevektor k fra matrisen bk
        X = Wk[k] @ Y + np.array([bk[:, k]] * I).transpose()
        Y_out[k] = Y + h * sigma(X)
        Y = Y_out[k]
    return Y_out

###################

def alt_gradient(wk, bk, w, mu, y, c):
    j_mu = d_eta(np.transpose(np.transpose(y[-1])@ w + mu * one)) @ (Z(y[-1], w, mu) - c)
    j_omega = y[-1] @ ((Z(y[-1], w, mu) - c) * d_eta(np.transpose(np.transpose(y[-1])@ w + mu * one)))
    p_k = np.outer(w, (np.transpose((Z(y[-1], w, mu) - c) * d_eta(np.transpose(np.transpose(y[-1])@ w + mu * one)))))
    arr_Pk = np.array([p_k])
    for i in range(1, K):
        p_k_min = np.array([np.array(arr_Pk[-i] + h * np.transpose(wk[-i-1]) @ (d_sigma(wk[-i-1]@y[-i-1] + np.transpose(np.array([bk[:, -i-1]]*I))) * arr_Pk[-i]))])
        arr_Pk = np.vstack((p_k_min, arr_Pk))
    j_wk = np.array([h*(arr_Pk[1] * (d_sigma(wk[0] @ y[0] + np.transpose(np.array([bk[:, 0]]*I))))) @ np.transpose(y[0])])
    j_bk = np.array(h*(arr_Pk[1] * (d_sigma(wk[0] @ y[0] + np.transpose(np.array([bk[:, 0]]*I))))) @ one)
    for j in range(1, K-1):
        j_wk_plus = np.array([h*(arr_Pk[j+1] * (d_sigma(wk[j] @ y[j] + np.transpose(np.array([bk[:, j]]*I))))) @ np.transpose(y[j])])
        j_wk = np.vstack((j_wk, j_wk_plus))
        j_bk_plus = np.array(h*(arr_Pk[j+1] * (d_sigma(wk[j] @ y[j] + np.transpose(np.array([bk[:, j]]*I))))) @ one)
        j_bk = np.vstack((j_bk, j_bk_plus))

    return j_wk, j_bk, j_omega, j_mu

####################

def optimering(grad_U, U_j):  # Returnerer de oppdaterte parameterene for neste iterasjon
    tau = [.1, .01]
    for i in range (K - 1):
        U_j[0][i] = U_j[0][i] - tau[0] * grad_U[0][i]
        U_j[1][:, i] = U_j[1][:, i] - tau[0] * grad_U[1][i]
    U_j[2] = U_j[2] - tau[0] * grad_U[2]
    U_j[3] = U_j[3] - tau[0] * grad_U[3]
    return U_j

#Her er gradient _J_U listen med J derivert på div element. U_j inneholder [W_K, B_k, w, my]

def adam_decent(gradient_J_U, U_j, j):              #her må man ta inn j som teller iterasjonstallet
    beta_1 = 0.9
    beta_2 = 0.999
    alpha = 0.01
    litn_epsilon = 1E-8
    for u in range(0, len(U_j)):                            #Tanken er å bevege seg gjennom de 4 variablene i U_j
        v_j = 0                                 # Dette er v_0
        m_j = 0                                 #dette er m_0
        g_j = gradient_J_U[u]  # Gradienten er en matrise
        m_j = beta_1 * m_j + (1 - beta_1) * g_j  # antar m_j i likn nå er m_j fra forrige iterasjon
        v_j = beta_2 * v_j + (1 - beta_2) * (g_j * g_j)  # siste gj ledd er matrise mult.
        m_j_hatt = m_j / (1 - beta_1 ** j)
        v_hatt = v_j / (1 - beta_2 ** j)
        if u == 0:                 #befinner seg i W_k når u = 0, og Bk når u = 1
            for i in range(K-1):                               #Itererer gjennom alle de k ulike matrisene eller bk verdiene
                U_j[u][i] = U_j[u][i] - (alpha * (m_j_hatt / (np.sqrt(v_hatt) + litn_epsilon)))[i]
        if u == 1:
            for p in range(K-1):
                U_j[1][:, p] = U_j[1][:, p] - (alpha * (m_j_hatt / (np.sqrt(v_hatt) + litn_epsilon)))[p]
        elif u == 2 or u == 3:
            U_j[u] = U_j[u] - alpha*(m_j_hatt/(np.sqrt(v_hatt)+litn_epsilon))       #U_jp1 = U_(j+1)
    return U_j


#####################

def algoritme(N, grad, y0=y0, C=C, K=K, sigma=sigma, h=h, Wk=Wk, bk=bk, w=w, mu=mu, C1 = C1):
    j = 0
    arr_z = np.array(Z(y0[0], w, mu))
    while j < N:
        C, Y0 = C, y0
        Yk = YK (Y0, K, sigma, h, Wk, bk)  # Array med K Yk matriser, kjører bildene igjennom alle lagene ved funk. YK
        d_Wk, d_bk, d_w, d_mu = grad(Wk, bk, w, mu, Yk, C)  # Regner ut gradieinten for parametrene våre
        Wk, bk, w, mu = optimering ([d_Wk, d_bk, d_w, d_mu], [Wk, bk, w, mu])  # Oppdaterer parametrene vhp. u_j
        j += 1
        np.append(arr_z, Z(Yk[-1], w, mu))
    return Yk, Wk, bk, w, mu, arr_z


def adams_algoritme(N, grad, y0=y0, C=C, K=K, sigma=sigma, h=h, Wk=Wk, bk=bk, w=w, mu=mu, C1 = C1):
    j = 0
    arr_z = np.array (Z(y0, w, mu))
    while j < N:

        j += 1
        C, Y0 = C, y0
        Yk = YK(Y0, K, sigma, h, Wk, bk)  # Array med K Yk matriser, kjører bildene igjennom alle lagene ved funk. YK
        d_Wk, d_bk, d_w, d_mu = grad(Wk, bk, w, mu, Yk, C)  # Regner ut gradieinten for parametrene våre
        Wk, bk, w, mu = adam_decent([d_Wk, d_bk, d_w, d_mu], [Wk, bk, w, mu], j)  # Oppdaterer parametrene vhp. u_j
        np.append(arr_z, Z(Yk[-1], w, mu))

    return Yk, Wk, bk, w, mu, arr_z


##########

def foreward_function(N):
    #algoritme (N, grad=alt_gradient, y0=y0, C=C, K=K, sigma=sigma, h=h, Wk=Wk, bk=bk, w=w, mu=mu, C1=C1)
    return sum(Z(x[0][-1], w, mu))/200

########

def lagre_array(Y_K, Wk, bk, w, mu, name): # lagrer alle verdiene fra læringsprosessen
    np.save('data/'+name+'.npy', [Y_K, Wk, bk, w, mu])

def loader(name):
    x = np.load (name+'.npy', allow_pickle=True)
    return x

###########



#Y_K, Wk, bk, w, mu = algoritme(40000, gradient, y0, C, K, sigma, h, Wk, bk, w, mu)
Y_K, Wk, bk, w, mu, arr_z = adams_algoritme(30000, alt_gradient, y0, C, K, sigma, h, Wk, bk, w, mu)

#lagre_array(Y_K, Wk, bk, w, mu, "40k_iterasoner_blårød")
# = loader("br_40k")



plot_progression(Y_K, np.transpose(C1))
#plot_model(foreward_function, x[0][0], np.transpose(C1), 1)
#plot_separation( , Y_K[-1], np.transpose(C1), 200)
