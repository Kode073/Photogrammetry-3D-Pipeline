import numpy as np
import image 




# Transforme les cordonnées de nc en matrice 3x3 tel quel soit comme dans l'équation 18
def matrice_n (nc) :
    res = np.zeros((3,3))
    res[1,0] = 1
    res[0,1] = -1
    res[0,2] = nc[1]
    res[2,0] = -nc[1]
    res[2,1] = nc[0]
    res[1,2] = -nc[0]
    return res

#Calcule la matrice P grâce aux points de calibrations
def syst (lxc,lnc,nb_points=True) :
    """ 
        input :
        lxc : liste de veteur de taille (1,4) : liste des coordonnées homogènes des points de calibrations irl
        lnc : liste de vecteur de taille (1,2) : liste des coordonnées non homogène des points sur l'image des points de calibrations
          
        output :
        p : matrice 3x4
    """
    
    if len(lxc)<6 and  nb_points:
        return "Pas assez de point"
    p = np.zeros((3,4))
    n = len(lxc)
    w = -1
    systeme = np.kron(matrice_n(lnc[0]),lxc[0])[:2]
    for i in range (1,n) :
        c = np.kron(matrice_n(lnc[i]),lxc[i])[:2]
        systeme = np.block([[systeme],[c]])
    U, S, Vh = np.linalg.svd(systeme, full_matrices=True)
    for i in range (3) :
        for j in range (4) :
            w = w+1
            p[i,j] = Vh.transpose()[:,-1][w]
    return -p

#transforme coordonnée euclidienne en coordonnée homogène
def homogene (x,t=1) :
    c = np.append(x,1)
    return c

#transforme une liste de coordonnée euclidien 
def homogene_l (l) : 
    l2 = []
    for i in range (len(l)) :
        l2.append(homogene(l[i]))
    return np.array(l2)




if __name__=="__main__":
    p = np.zeros((3,4))
    vectp = np.zeros((12,1))

    xct = np.ones((1,4))
    nc = np.ones((3,3))


    c13 = np.kron(nc,xct)[:2]
    c = np.kron(nc,xct)[:2]
    for i in range (5) :
        c = np.block([[c],[c13]])
    #Test
    liste_Xc=np.array([[-100,100,100],[0,100,0],[100,100,100],[100,0,0],[100,-100,100],[0,-100,0],[-100,-100,100],[-100,0,0]])
    liste_nc=np.array([[872,3800],[2330,2842],[2524,2016],[2330,1190],[872,232],[1254,1407],[140,2016],[1254,2625]])


    print(syst(homogene_l(liste_Xc),liste_nc))

    
#a,b,c =image.image(0,-45,-45,-20,0,330,affichage=False)
#print(c)


