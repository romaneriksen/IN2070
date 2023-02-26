from imageio import imread
import imageio
from matplotlib import pyplot as plt
import numpy as np
from numpy import r_
from numba import jit
from PIL import Image  
import PIL  

@jit
def main():

    fil = 'uio.png'
    img = imread(fil, as_gray=True)
    N,M = img.shape

    # Kvantifiseringsmatrise 
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    q = [0.1, 0.5, 2, 8, 32]
    entropier = [0] * 5

    cos_image = 0
    image_reconstructed = 0
    indeks = 0

    # For-loop som lagrer de komprimerte bildene og utfører entropi-beregning 
    for value in q: 
        cos_image = cos_transform(img, Q, value)
        image_reconstructed = inverse_cos_transform(cos_image, Q, value)
        entropier[indeks] = entropi(cos_image)
        indeks += 1
        # string = "uio_q_" + str(value) + ".jpeg" 
        # imageio.imwrite(string, image_reconstructed)

    for value in entropier:
        print("Entropi: ", value)
    

    # Foor-loop som programmatisk verifiserer at rekonstruert bilde er likt originalt
    # img = img.astype('float64')
    # image_reconstructed = image_reconstructed.astype('float64')
    # sjekk = True
    # for x in range(N):
    #     for y in range(M):
    #         if (img[x,y]) == (image_reconstructed[x,y]):
    #             pass
    #         else:
    #             sjekk = False
    # print(sjekk)


# Funksjon som regner ut entropi for transformnert bilde. 
def entropi(img):
    dict = {}
    N,M = img.shape
    for x in range(N):
        for y in range(M):
            if img[x,y] in dict:
                dict[img[x,y]] += 1
            else:
                dict[img[x,y]] = 1
    
    for element in dict:
        dict[element] = dict[element]/(N*M)

    entropi = 0
    for element in dict:
        entropi += dict[element] * np.log2(1/dict[element])

    return entropi


# Funksjon som rekonstruerer det komprimerte bildet
@jit
def inverse_cos_transform(img, Q, q):

    Q = q*Q

    img_rec = np.zeros(np.shape(img))
    N,M = img.shape
    blokk = np.zeros((8,8))

    # For-loops som henter ut hver 8x8 blokk i det transformerte bildet
    for i in range(0,N,8):
        for j in range(0,M,8):
            blokk = img[i:i+8,j:j+8]
            Y,X = blokk.shape
            blokk = np.around(blokk * Q)

            # For-loops som tar seg av indekseringen for resultatbildet
            for u in range(X):
                for v in range(Y):
                    img_rec[i+u,j+v] = idct(blokk, u, v)  
    return img_rec + 128

# Funksjon som utfører cosinus-transformen på originalbildet
@jit
def cos_transform(img, Q, q):

    Q = q*Q

    img = img - 128 
    img_cos = np.zeros(np.shape(img))
    N,M = img.shape
    blokk = np.zeros((8,8))

    # For loop som henter ut hver 8x8 blokk fra originalbildet
    for i in range(0,N,8):
        for j in range(0,M,8):
            blokk = img[i:i+8,j:j+8]
            Y,X = blokk.shape

            # For-loops som tar seg av indekseringen for det transformerte bildet
            for u in range(X):
                for v in range(Y):
                    img_cos[i+u,j+v] = np.around(dct(blokk, u, v) / Q[u,v])
    return img_cos

# Funksjon som regner ut invers DCT  
@jit
def idct(img, x, y):
    sum = 0
    for u in range(8):
        for v in range(8):
            verdi = c(u) * c(v) * img[u,v] * cos(x,u) * cos(y,v)
            sum = sum + verdi
    sum *= 0.25 
    return sum

# Funksjon som regner ut DCT på originalbildet
@jit
def dct(img, u, v):
    sum = 0
    for x in range(8):
        for y in range(8):
            verdi = img[x,y] * cos(x,u) * cos(y,v)
            sum = sum + verdi
            pass
    sum *= 0.25 * c(u) * c(v)
    return sum

# Funksjon som regner ut cos verdiene i DCT funksjonen
@jit
def cos(a,b):
    ret = np.cos(((2*a+1)*b*np.pi)/16)
    return ret

# Funksjon for c
@jit
def c(tall):
    if tall == 0:
        return 1/(np.sqrt(2))
    else:
        return 1

if __name__ == main():
    main()