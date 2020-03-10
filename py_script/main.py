import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
import loader as ld
import spirals as sp
from plotting import plot_progression, plot_model, plot_separation
import random as rn

rng = np.random.randn

images, label = ld.get_dataset("training", 3, 6, "../mnist")
images = images/255

K = 15  # Antall lag
d = 784  # Antall piksel-elementer til hvert bilde. Hvert bilde er stablet opp i en vektor av lengde d
I = 70  # Antall bilder
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


def big_j(z, c): return 0.5*(la.norm(z-c))**2
"""
def big_j(Z, c):            #Fungerer for numpy arrays
    c = -1*c
    big_j = 0.5*la.norm(np.add(Z, c))**2
    return big_j
"""
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

def alt_gradient(wk, bk, w, mu, y, c, K):
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

def optimering(grad_U, U_j, K):  # Returnerer de oppdaterte parameterene for neste iterasjon
    tau = [.05, .01]
    for i in range (K - 1):
        U_j[0][i] = U_j[0][i] - tau[0] * grad_U[0][i]
        U_j[1][:, i] = U_j[1][:, i] - tau[0] * grad_U[1][i]
    U_j[2] = U_j[2] - tau[0] * grad_U[2]
    U_j[3] = U_j[3] - tau[0] * grad_U[3]
    return U_j

#Her er gradient _J_U listen med J derivert på div element. U_j inneholder [W_K, B_k, w, my]

def adam_decent(gradient_J_U, U_j, j, K):              #her må man ta inn j som teller iterasjonstallet
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


def adams_algoritme(N, grad, y0, K, C, h, Wk, bk, w, mu, sigma=sigma):
    j = 0
    arr_z = np.array (Z(y0, w, mu))
    arr_j = np.random.rand(N)
    while j < N:
        C, Y0 = C, y0
        Yk = YK(Y0, K, sigma, h, Wk, bk)  # Array med K Yk matriser, kjører bildene igjennom alle lagene ved funk. YK
        d_Wk, d_bk, d_w, d_mu = grad(Wk, bk, w, mu, Yk, C, K)  # Regner ut gradieinten for parametrene våre

        z = Z(Yk[-1], w, mu)
        arr_j[j] = big_j (z, C)
        j += 1

        Wk, bk, w, mu = adam_decent([d_Wk, d_bk, d_w, d_mu], [Wk, bk, w, mu], j, K)  # Oppdaterer parametrene vhp. u_j
        np.append(arr_z, Z(Yk[-1], w, mu))

        print(j,"faaaaaaak",len(arr_j))
        if (j % 10) == 0:
            print(j)

    return Yk, Wk, bk, w, mu, arr_z, arr_j

##########


#Det er omtrent 12000 bilder av treningsutvalget som har siffrene vi ønsker å trene etter.
#her bestemmer
def numb_req(N, grad, K, images, labels, ant_sett, ant_i_sett=I):
    Wk = rng (K, d, d)
    w = rng (d)
    mu = rng (1)
    bk = rng (d, K)

    #Generer rand. tall mellom (0, 12000-ant_i_sett*ant_sett
    rand_tall = np.random.randint(0, len(images[1])-(ant_sett*ant_i_sett))
    arr_big_j = np.random.rand(ant_sett, N)
    for i in range(ant_sett):
        print("starter et sett")
        Yk = np.random.rand(d, ant_i_sett)        # dette er y0 for hver iterasjon
        korr_tall = np.random.rand(ant_i_sett)
        for j in range(1, rand_tall+ ant_i_sett*i, rand_tall+ant_i_sett + ant_i_sett*i-1):
            Yk[:, j % rand_tall-1] = images[:, j-1]   # fyller inn bilder for valgt intervall
            korr_tall[j % rand_tall - 1] = labels[j]
        print("bildeutvalg funnet")
        Yk, Wk, bk, w, mu, arr_z, arr_j = adams_algoritme(N, grad, Yk, K, korr_tall, 0.1, Wk, bk, w, mu)
        arr_big_j[i] = arr_j
        print(i)

    return Yk, Wk, bk, w, mu, arr_z, arr_big_j


Yk, Wk, bk, w, mu, arr_z, arr_big_j = numb_req(20, alt_gradient, 10, images, label, 2)

print(arr_big_j)
x = np.linspace(1, 20, 20)

plt.figure()
plt.plot(x, arr_big_j[0], '.')
plt.plot(x, arr_big_j[1], '.')
plt.show()




########

def lagre_array(Y_K, Wk, bk, w, mu, name): # lagrer alle verdiene fra læringsprosessen
    np.savez('data/'+name+'.npy', [Y_K, Wk, bk, w, mu])

def loader(name):
    x = np.load (name+'.npy', allow_pickle=True)
    return x

###########


def forward_function(x, I=I):       # Kjører gjennom algoritmen EN gang med riktig
    N = 1                      # tilpasning for plottefunksjonen
    I **=2
    _, C1 = sp.get_data_spiral_2d(I)
    C = np.reshape(C1, I)
    print("C: ",np.shape(C))
    return algoritme(N, grad=gradient, y0 = x, C=C, K=K, sigma=sigma, h=h, Wk=Wk, bk=bk, w=w, mu=mu, C1=C1, I=I)[5]

def last_function(x, w=w, mu=mu,I=I):
    return eta(np.transpose(x) @ w + mu * np.ones(I**2))  # x = siste Y_K)



#Fungerer best for mange iterasjoner:
Y_K, Wk, bk, w, mu, arr_z = algoritme(60000, gradient, y0, C, K, sigma, h, Wk, bk, w, mu)
plot_progression(Y_K, np.transpose(C1))
plot_model(forward_function, y0, np.transpose(C1), I)
plot_separation(last_function,Y_K[-1,:,:],C1,I)
