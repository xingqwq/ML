# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

# %%
def makeData(numScale,mean,sigma):
    data = np.random.multivariate_normal(mean,sigma,numScale)
    return np.array(data)

# %%
def PCASolver(data,dim):
    mu = (np.sum(data,axis=0)/len(data))
    dataCen = data-mu
    val,vec = np.linalg.eig(dataCen.T@dataCen)
    index = np.argsort(val)
    index = index[max(0,len(index)-dim):][::-1]
    result = []
    for i in index:
        result.append(vec[:,i]) #这里要提取列向量
    result = np.array(result)
    return dataCen@result.T@result + mu

def Trans2DTo1D(numScale,mean,sigma):
    data = makeData(numScale,mean,sigma)
    dataPCA = PCASolver(data[:],1)
    plt.scatter(data[0:,0],data[0:,1],s=5,c='blue')
    plt.scatter(dataPCA[0:,0],dataPCA[0:,1],s=5,c='red')
    plt.plot(dataPCA[0:,0],dataPCA[0:,1],c='black')
    plt.show()
    
def Trans3DTo2D(numScale,mean,sigma):
    data = makeData(numScale,mean,sigma)
    dataPCA = PCASolver(data[:],2)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[0:,0],data[0:,1],data[0:,2],c='blue')
    ax.scatter(dataPCA[0:,0],dataPCA[0:,1],dataPCA[0:,2], c="red")
    ax.plot_trisurf(dataPCA[0:,0],dataPCA[0:,1],dataPCA[0:,2], color="green", alpha=0.3)
    plt.show()

def calcPSNR(ori, dataPCA):
    print(np.mean((ori - dataPCA) ** 2))
    print(ori.shape)
    return 10 * np.log10(np.max(ori)**2 / np.mean((ori - dataPCA) ** 2))

def ImgPCA(dimList):
    grayImg = cv2.imread("psc.png", cv2.IMREAD_GRAYSCALE)
    data = np.asarray(grayImg)
    plt.subplot(2, 3, 1)
    plt.axis("off")
    plt.imshow(data)
    for i in range(0,len(dimList)):
        dataPCA = PCASolver(data[:],dimList[i])
        plt.subplot(2, 3, i+2)
        plt.axis("off")
        plt.title("Dim:"+str(dimList[i])+" PSNR:{:.4f}".format(calcPSNR(data,dataPCA)))
        plt.imshow(dataPCA)
    plt.show()
# %%
Trans2DTo1D(50,[3,6],[[5,0],[0,0.2]])
Trans3DTo2D(50,[3,7,5],[[5,0,0],[0,5,0],[0,0,0.2]])
ImgPCA([140,120,40,10,5])

