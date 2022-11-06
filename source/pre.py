import torchvision.transforms as transforms
import numpy as np
def pre_processing (data, size = 56):
    #grayscale
    transformGrey = transforms.Grayscale()
    dataGrey = transformGrey(data)
    # reducing the size of the images
    transformResize = transforms.Resize(size)
    dataResized = transformResize(dataGrey)
    #normalize
    dataPre = dataResized/255
    dataPre = np.array(dataPre.reshape(len(dataPre),size,size))

    return dataPre


        
