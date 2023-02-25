from imageio import imread
from matplotlib import pyplot as plt
import numpy as np
import math

def main():

    fil = 'portrett.png' # portrettbildet
    img = imread(fil, as_gray=True, )
    plt.imshow(img,cmap='gray')
    #plt.title("Portrett.png før transform")

    fil2 = 'geometrimaske.png' # geometrimasken
    img2 = imread(fil2, as_gray=True)

    sd_new = 64
    mv_new = 127

    portrett_transformert = graaToneTransform(sd_new, mv_new, img)

    forward_mapping(portrett_transformert, img2)
    backward_mapping_nabo(portrett_transformert, img2)
    backward_mapping_interpolasjon(portrett_transformert, img2)

def graaToneTransform(sd_new, mv_new, img):


    sd_old = np.std(img) # gammelt standardavvik
    print(sd_old)
    mv_old = np.median(img) # gammel middelverdi
    print(mv_old)

    # gråtonetransform
    a = np.sqrt((sd_new**2)/(sd_old**2))
    b = mv_new - (a*mv_old)

    N,M = img.shape
    img_out = np.zeros((N,M))
    #img_out = []
    for i in range(N):
        for j in range(M):
            img_out[i,j] = a*img[i,j]+b
            #img_out.append(img[i,j])

    sd_transform = np.std(img_out)
    mv_transform = np.median(img_out)
    print("Standardavvik etter transform: ",sd_transform,"\nMiddelverdi etter transform: ",mv_transform) 

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Portrett.png før transform")

    plt.subplot(1, 2, 2)
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.title("Portrett.png etter transform")

    #plt.savefig('portrett.ny.png')
    plt.show()
    return img_out

def forward_mapping(img, img2):

    A = np.array([[83, 88, 1], [119, 67, 1], [128, 107, 1]]) # Array for a0, a1, a2
    b = np.array([168, 341, 256]) # Hvilke verdier x_mark skal ha

    C = np.array([[83, 88, 1], [119, 67, 1], [128, 107, 1]]) # Array for b0, b1, b2
    d = np.array([257, 257, 440]) # Hvilke verdier y_mark skal ha

    sol = np.linalg.solve(A, b) # Løser for a0, a1 , a2
    sol2 = np.linalg.solve(C, d) # løser for b0, b1 , b2

    H = np.array([[sol[0], sol[1], sol[2]],
                  [sol2[0], sol2[1], sol2[2]]])

    N,M = img.shape # Dimensjoner for portrett
    print(N)
    print(M)

    B,A = img2.shape # Dimensjoner for utbilde

    img_out = np.zeros((B,A))

    a0 = H[0][0]
    a1 = H[0][1]
    a2 = H[0][2]
    b0 = H[1][0]
    b1 = H[1][1]
    b2 = H[1][2]

    for x in range(M):
        for y in range(N):
            x_mark = round((a0*x) + (a1*y) + a2)
            y_mark = round((b0*x) + (b1*y) + b2)
                
            x_mark = int(x_mark)
            y_mark = int(y_mark)
            if x_mark in range(A) and y_mark in range(B):
                img_out[y_mark,x_mark] = img[y,x]

    """
    plt.subplot(1, 3, 1)
    plt.imshow(img,cmap='gray')
    plt.title("Portrett.png før forward mapping")
    """

    plt.subplot(1, 2, 1)
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.title("Portrett.png etter forward mapping")

    plt.subplot(1,2,2)
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255, alpha = 0.12)
    
    plt.savefig('portrett_forlengs.png')

    plt.show()
  
def backward_mapping_nabo(img, img2):

    A = np.array([[83, 88, 1], [119, 67, 1], [128, 107, 1]]) # Array for a0, a1, a2
    b = np.array([168, 341, 256]) # Hvilke verdier x_mark skal ha

    C = np.array([[83, 88, 1], [119, 67, 1], [128, 107, 1]]) # Array for b0, b1, b2
    d = np.array([257, 257, 440]) # Hvilke verdier y_mark skal ha

    sol = np.linalg.solve(A, b) # Løser for a0, a1 , a2
    sol2 = np.linalg.solve(C, d) # løser for b0, b1 , b2

    H = np.array([[sol[0], sol[1], sol[2]],
                  [sol2[0], sol2[1], sol2[2]],
                  [0 ,0 ,1]])

    H_i = np.linalg.inv(H)

    N,M = img.shape # Dimensjoner for portrett
    print(N)
    print(M)

    B,A = img2.shape # Dimensjoner for utbilde
    print(A)
    print(B)

    img_out = np.zeros((B,A))

    a0 = H_i[0][0]
    a1 = H_i[0][1]
    a2 = H_i[0][2]
    b0 = H_i[1][0]
    b1 = H_i[1][1]
    b2 = H_i[1][2]

    for x_mark in range(A):
        for y_mark in range(B):

            x = round((a0*x_mark) + (a1*y_mark) + a2)
            y = round((b0*x_mark) + (b1*y_mark) + b2)

            x = int(x)
            y = int(y)
            if x in range(M) and y in range(N):
                #img_out[x_mark,y_mark] = img[y,x]
                img_out[y_mark,x_mark] = img[y,x]
            else:
                #img_out[x_mark,y_mark] = 0
                img_out[y_mark,x_mark] = 0

    """
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Portrett.png før baklengs-mapping")
    """

    plt.subplot(1, 2, 1)
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.title("Portrett.png etter baklengs-mapping")

    plt.subplot(1,2,2)
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255, alpha = 0.5)

    plt.savefig('portrett.baklengs.png')

    plt.show()

def backward_mapping_interpolasjon(img, img2):

    A = np.array([[83, 88, 1], [119, 67, 1], [128, 107, 1]]) # Array for a0, a1, a2
    b = np.array([168, 341, 256]) # Hvilke verdier x_mark skal ha

    C = np.array([[83, 88, 1], [119, 67, 1], [128, 107, 1]]) # Array for b0, b1, b2
    d = np.array([257, 257, 440]) # Hvilke verdier y_mark skal ha

    sol = np.linalg.solve(A, b) # Løser for a0, a1 , a2
    sol2 = np.linalg.solve(C, d) # løser for b0, b1 , b2

    H = np.array([[sol[0], sol[1], sol[2]],
                  [sol2[0], sol2[1], sol2[2]],
                  [0 ,0 ,1]])

    H_i = np.linalg.inv(H)

    N,M = img.shape # Dimensjoner for portrett

    B,A = img2.shape # Dimensjoner for utbilde
    
    img_out = np.zeros((B,A))

    a0 = H_i[0][0]
    a1 = H_i[0][1]
    a2 = H_i[0][2]
    b0 = H_i[1][0]
    b1 = H_i[1][1]
    b2 = H_i[1][2]

    for x_mark in range(A):
        for y_mark in range(B):

            x = (a0*x_mark) + (a1*y_mark) + a2
            y = (b0*x_mark) + (b1*y_mark) + b2

            #x = int(x)
            #y = int(y)
            x_0 = math.floor(x)
            x_1 = math.ceil(x)
            y_0 = math.floor(y)
            y_1 = math.ceil(y)

            sum_x = x - x_0
            sum_y = y - y_0
            
            p = img[y_0, x_0] + (img[y_0, x_1] - img[y_0, x_0]) * sum_x
            q = img[y_1, x_0] + (img[y_1, x_1] - img[y_1, x_0]) * sum_x
            img_out[y_mark,x_mark] = p + (q - p) * sum_y
    
    """
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Portrett.png før bilineær interpolasjon")
    """

    plt.subplot(1, 2, 1)
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.title("Portrett.png etter bilineær interpolasjon")

    plt.subplot(1,2,2)
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255, alpha = 0.5)

    plt.savefig('portrett.interpolasjon.png')

    plt.show()

if __name__ == main():
    main()