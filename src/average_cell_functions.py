import numpy as np

def extractMeasureFromImage(img,zeroThresh=1E-14):
    dim=img.shape

    pos=np.zeros(dim+(2,),dtype=np.double)
    pos[:,:,0]=np.arange(dim[0]).reshape((-1,1))
    pos[:,:,1]=np.arange(dim[1]).reshape((1,-1))
    pos=pos.reshape((-1,2))
    keep=(img.ravel()>zeroThresh)
    mu=img.ravel()[keep]
    pos=pos[keep]

    return (mu,pos)