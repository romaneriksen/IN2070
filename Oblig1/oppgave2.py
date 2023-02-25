from imageio import imread
from matplotlib import pyplot as plt
import numpy as np
import math
from numba import jit


def main():

    fil = 'cellekjerner.png' # bildet av cellekjerner
    img = imread(fil, as_gray=True)
    plt.imshow(img, cmap='gray')

    filt = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    canny(img, 7, 60, 30)

    
def canny(image, sigma, t_s, t_w):

    gauss = gauss_filter(sigma)

    blur = convolve(image, gauss)

    output = gradient_magnitude(blur)

    a = tynning(output[0], output[1])

    t_s = 60
    t_w = 30
    b = hysterese(a, t_s, t_w)

    #plt.subplot(1, 2, 1)
    #plt.imshow(img, cmap='gray')

    #plt.subplot(1, 2, 2)
    plt.imshow(b, cmap='gray', vmin = 0, vmax = 255)

    tittel = "Sigma: ",sigma, ", T_s: ",t_s, " T_w: ",t_w

    plt.title(str(tittel))

    plt.savefig('Detekterte_kanter.png')
    
    plt.show()

    pass


@jit
def hysterese(image, t_s, t_w):

    img_s = image
    img_w = image
    img_n = image

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            if (image[i,j] >= t_s):
                img_s[i,j] = True

            elif (t_w <= image[i,j] < t_s):
                img_w[i,j] = True
                
            else:
                img_s[i,j] = False
                img_w[i,j] = False
    
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):

            if (img_s[i,j]):
                img_s[i,j] = 255

            if (((img_s[i+1, j-1]) or (img_s[i+1, j]) or (img_s[i+1, j+1])
                or (img_s[i, j-1]) or (img_s[i, j+1]) or (img_s[i-1, j-1]) 
                or (img_s[i-1, j]) or (img_s[i-1, j+1])) and (img_w[i,j])):
                    img_s[i, j] = 255

            else:
                img_s[i, j] = 0
    
    return img_s
  
@jit
def tynning(image, alpha_xy):
    
    M, N = image.shape
    output = np.zeros((M,N))
    theta_dot = np.zeros((M,N))

    for i in range(M):
        for j in range(N):
            theta_dot[i,j] = round(((alpha_xy[i,j])*(180/np.pi)*(1/45)*45)) % 180
    
    nabo_l = 0
    nabo_r = 0

    for x in range(1,M-1):
        for y in range(1,N-1):

            if (0 <= theta_dot[x,y] < 22.5) or (157.5 <= theta_dot[x,y] <= 180):
                nabo_l = image[x,y-1]
                nabo_r = image[x,y+1]
            
            elif (22.5 <= theta_dot[x,y] < 67.5):
                nabo_l = image[x-1,y+1]
                nabo_r = image[x+1,y-1]

            elif (67.5 <= theta_dot[x,y] < 112.5):
                nabo_l = image[x-1,y]
                nabo_r = image[x+1,y]

            elif (112.5 <= theta_dot[x,y] < 157.5):
                nabo_l = image[x-1,y-1]
                nabo_r = image[x+1,y+1]

            if (nabo_l <= image[x,y]) and (nabo_r <= image[x,y]):
                output[x,y] = image[x,y]
            else:
                output[x,y] = 0

    return output

def gauss_filter(sigma):

    size = math.ceil(1 + 8*sigma) / 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    A = 1 / (2 * np.pi * sigma**2)
    gauss = np.exp(-((x**2 + y**2) / (2*sigma**2))) * A
    return gauss

@jit
def gradient_magnitude(image):

    sobel_x = np.array([[-1, 0, 1], [-2, 0 , 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    g_x = convolve(image, sobel_x)
    g_y = convolve(image, sobel_y)

    G_xy = np.sqrt(g_x**2 + g_y**2)
    alpha_xy = np.arctan2(g_y, g_x)

    return G_xy, alpha_xy

@jit
def convolve(image, filter):

    filter = np.flipud(np.fliplr(filter))

    imagePadded = pad(image, filter)

    xFilterShape = filter.shape[0]
    yFilterShape = filter.shape[1]
    xImageShape = image.shape[0]
    yImageShape = image.shape[1]

    output = np.zeros((image.shape[0], image.shape[1]))
    
    for x in range(xImageShape):
        for y in range(yImageShape):
            output[x, y] = (filter * imagePadded[x: x + xFilterShape, y: y + yFilterShape]).sum()
            

    return output

def pad(image, filter):

    iH, iW = image.shape
    fH, fW = filter.shape

    x = math.floor(fW/2)
    y = math.floor(fH/2)

    paddedImage = image
    for i in range(y):
        paddedImage = pad_tb(paddedImage)
    for j in range(x):
        paddedImage = pad_lr(paddedImage)

    return paddedImage

def pad_tb(array):

    y, x = array.shape

    imagePadded = np.zeros((y+2, x))

    imagePadded[1:y+1, 0:x+1] = array
    imagePadded[0:1]= array[0]
    imagePadded[y+1:y+2]= array[y-1]

    return imagePadded

def pad_lr(array):

    y, x = array.shape

    imagePadded = np.zeros((y, x+2))
    imagePadded[0:y+1, 1:x+1] = array
    #print(imagePadded, "\n")
    for i in range(y):
        imagePadded[i][0] = array[i][0]
        imagePadded[i][x+1] = array[i][x-1]

    #print(imagePadded)
    return imagePadded

def padding(array):

    padding = 1

    y, x = array.shape # x = horisontal, y = vertikal

    new_array = np.zeros((y+2, x+2))
    
    imagePadded = np.zeros((array.shape[0] + padding*2, array.shape[1] + padding*2))
    imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = array

    print(imagePadded)
    
    top_row = array[0]
    bottom_row = array[y-1]

    new_top_row = np.zeros((top_row.size+2))
    new_bottom_row = np.zeros((bottom_row.size+2))

    k = 0
    for x in range(new_top_row.size):
        if x == 0:
            new_top_row[0] = top_row[0]
            new_bottom_row[0] = bottom_row[0]

        elif x == new_top_row.size - 1:
            new_top_row[x] = top_row[top_row.size-1]
            new_bottom_row[x] = bottom_row[bottom_row.size-1]

        else:
            new_top_row[x] = top_row[k]
            new_bottom_row[x] = bottom_row[k]
            k+=1

    imagePadded[0] = new_top_row
    imagePadded[imagePadded.shape[0]-1] = new_bottom_row

    for x in range(imagePadded.shape[0]):
        imagePadded[x][0] = imagePadded[x][1]
        imagePadded[x][imagePadded.shape[1]-1] = imagePadded[x][imagePadded.shape[1]-2] 

    print()
    print(imagePadded)
    return imagePadded

if __name__ == main():
    main()

