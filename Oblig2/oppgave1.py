from imageio import imread
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import math
from numba import jit
import time

def main():

    fil = 'cow.png' 
    img = imread(fil, as_gray=True)

    # plt.subplot(1,3,1)
    # plt.imshow(img, cmap='gray')
    # plt.title("cow.png")

    mv_filter_15 = np.ones((15,15)) / 15**2

    romlig_conv = signal.convolve2d(img, mv_filter_15) # Konvolusjon
    freq_filtered = freq_filtering(img, mv_filter_15) # Filtrering i frekevensdomenet

    plt.subplot(1,2,1)
    plt.imshow(romlig_conv, cmap='gray')
    plt.title("Romlig konvulsjon")

    plt.subplot(1,2,1)
    plt.imshow(freq_filtered.real, cmap='gray')
    plt.title("Filtrering i frekvensdomenet")
    plt.show()

    romlig_conv_same = signal.convolve2d(img, mv_filter_15, 'same') # Konvolusjon med 'same' paramtere
    freq_filtered_2 = freq_filtering_2(img, mv_filter_15) # Filtrering med implisitt padding

    plt.subplot(1,2,1)
    plt.imshow(romlig_conv_same, cmap='gray')
    plt.title("Romlig konvulsjon med 'same'")

    plt.subplot(1,2,2)
    plt.imshow(freq_filtered_2.real, cmap='gray')
    plt.title("Filtrering i frekvensdomenet implisitt padding")
    plt.show()

    runTime_1_3(img) # Funksjon for kjøretid
    pass

def runTime_1_3(img):

    antall_filtre = 30
    mv_filter = 0
    tider_convolve = [0] * antall_filtre

    # For-loop som utfører konvolusjon for 30 ulike middelverdifiltre
    for i in range(antall_filtre):
        start_tid = time.time ()

        mv_filter = np.ones((i+1,i+1)) / (i+1)**2
        romlig_conv = signal.convolve2d(img, mv_filter, 'same')
        
        stopp_tid = time.time() - start_tid
        tider_convolve[i] = stopp_tid

    tider_freq = [0] * antall_filtre

    # For-loop som utfører filtrering i frekevensdomenet for 30 ulike middelverdifiltre
    for i in range(antall_filtre):
        start_tid = time.time ()

        mv_filter = np.ones((i+1,i+1)) / (i+1)**2
        freq_filt = freq_filtering_2(img, mv_filter)
        
        stopp_tid = time.time() - start_tid
        tider_freq[i] = stopp_tid

    conv = plt.plot(list(range(1,31)), tider_convolve, label = "Convolve2D")
    freq = plt.plot(list(range(1,31)), tider_freq, label = "Frequency")
    leg = plt.legend(loc='upper center')
    plt.xlabel("Antall filtre")
    plt.ylabel("Tid")
    plt.show()

# Funksjon for filtrering i frekvensdomenet
def freq_filtering(img, filter):

    padded_mv_filter_15 = pad(img, filter) # funksjon som padder filteret til riktig størrelse

    img_fft = np.fft.fft2(img) # Transformasjon til frekvensdomenet

    mv_filter_shift = np.fft.ifftshift(padded_mv_filter_15)
    mv_filter_fft = np.fft.fft2(mv_filter_shift)

    freq_filt = img_fft * mv_filter_fft # punktvis multiplikasjon
    
    freq_filt = np.fft.ifft2(freq_filt) # invers transform tilbake til bildedomenet

    return freq_filt

# Funksjon for filtrering i bildedomenet med implisitt padding
def freq_filtering_2(img, filter):
    
    mv_filter_shift = np.fft.ifftshift(filter)
    mv_filter_fft = np.fft.fft2(mv_filter_shift, (768, 1024))
    img_fft = np.fft.fft2(img)

    freq_filtered_2 = mv_filter_fft * img_fft 
    freq_filtered_2 = np.fft.ifft2(freq_filtered_2)
    
    return freq_filtered_2


def pad(image, filter):

    # fH, fW = filter.shape
    # filter = np.ones((fH-1, fW-1))

    iH, iW = image.shape
    fH, fW = filter.shape

    x = math.ceil((iW-fW)/2)
    y = math.ceil((iH-fH)/2)

    paddedImage = filter
    for i in range(y):
        paddedImage = pad_tb(paddedImage)
    for j in range(x):
        paddedImage = pad_lr(paddedImage)

    return paddedImage

def pad_tb(array):

    y, x = array.shape

    if (y % 2 == 0):
        imagePadded = np.zeros((y+2, x))
    else:
        imagePadded = np.zeros((y+1, x))

    imagePadded[1:y+1, 0:x+1] = array
    imagePadded[0:1]= 0
    imagePadded[y+1:y+2]= 0


    return imagePadded

def pad_lr(array):

    y, x = array.shape

    if (x % 2 == 0):
        imagePadded = np.zeros((y, x+2))
        imagePadded[0:y+1, 1:x+1] = array
        for i in range(y):
            imagePadded[i][0] = 0
            imagePadded[i][x+1] = 0
    else:
        imagePadded = np.zeros((y, x+1))
        imagePadded[0:y+1, 1:x+1] = array
        for i in range(y):
            imagePadded[i][0] = 0
            imagePadded[i][x] = 0

    return imagePadded


if __name__ == main():
    main()





