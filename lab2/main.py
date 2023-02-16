# %%
import numpy as np
import matplotlib.pyplot as plt

class GradientDescentOptimizer:
    def __init__(self,numScale,learningRate,lossLimit):
        self.numScale=numScale
        self.learningRate=learningRate
        self.lossLimit=lossLimit

    def sigmoid(self,xMatrix,wVec):
        return 1 / (1 + np.exp(-np.dot(xMatrix, wVec)))
    
    def calcGradient(self,xMatrix,wVec,tVec,lambdaPenalty):
        return xMatrix.T.dot(tVec-self.sigmoid(xMatrix,wVec))+ lambdaPenalty * wVec
    
    def calcLoss(self,xMatrix,wVec,tVec,lambdaPenalty):
        return np.sum(-np.dot(tVec.T, np.log(self.sigmoid(xMatrix, wVec)))-np.dot((np.ones((len(tVec), 1)) - tVec).T,np.log(np.ones((len(tVec), 1)) - self.sigmoid(xMatrix, wVec)),))/len(xMatrix) + 0.5*lambdaPenalty*np.dot(wVec.T,wVec)
    
    def optimizeParam(self,xMatrix,wVec,tVec,lambdaPenalty):
        # 观测优化过程以及优化次数的限制
        ck=0
        # 记录Loss值
        lastLoss=self.calcLoss(xMatrix,wVec,tVec,lambdaPenalty)
        loss=lastLoss
        while True:
            wVecTmp=wVec+self.learningRate*self.calcGradient(xMatrix,wVec,tVec,lambdaPenalty)
            lossTmp=self.calcLoss(xMatrix,wVecTmp,tVec,lambdaPenalty)
            if lossTmp>lastLoss:
                self.learningRate*=0.5
            else:
                wVec=wVecTmp
                loss=lossTmp
                if(np.abs(loss-lastLoss)<self.lossLimit) and (loss<=1.3 or wVec.shape[0]<=3):
                    break
                lastLoss=loss
            ck+=1
        return wVec,ck
# 生成数据
def makeData(numScale,mean,sigma):
    data = np.random.permutation(np.random.multivariate_normal(mean,sigma,numScale))
    data = np.hstack((np.array([1 for _ in range(0,len(data))]).reshape(len(data),1),data))
    # Shuffle数据，并划分数据集和测试集
    train= data[0:int(0.85*numScale),]
    test= data[len(train):len(data),]
    return train,test

# 画出划分线线
def drawLine(wVec):
    x = np.linspace(-2, 8, 100)
    y = (-wVec[0][0] - wVec[1][0] * x) / wVec[2][0]
    plt.plot(x,y,label="划分线")

# 计算正确率
def judge(wVec,test1,test2):
    result1 = test1.dot(wVec)
    result2 = test2.dot(wVec)
    errCnt = 0
    for i in result1:
        if i > 0:
            errCnt += 1
    for i in result2:
        if i < 0:
            errCnt += 1
    return (len(test1)+len(test2)-errCnt)/(len(test1)+len(test2))

# %%
# 超参
NUM_SCALE = 1000
LEARNING_RATE = 0.01
LOSS_LIMIT = 1e-8

# 生成数据
train1,test1 = makeData(NUM_SCALE,np.array([1,4]),np.array([[2, 3], [3, 2]]))
train2,test2 = makeData(NUM_SCALE,np.array([4,7]),np.array([[2, 3], [3, 2]]))

# %%
# 生成训练数据并无正则项的梯度下降
data = np.vstack((train1,train2))
label = np.vstack((np.zeros((len(train1),1)),np.ones((len(train2),1))))
optimizer=GradientDescentOptimizer(NUM_SCALE,LEARNING_RATE,LOSS_LIMIT)
wVec,cnt = optimizer.optimizeParam(data,np.ones((3,1)),label,1e-6)

# 显示结果
print("正确率："+str(judge(wVec,test1,test2)))
drawLine(wVec)
plt.title("NUM_SCALE:"+str(NUM_SCALE)+" 迭代次数："+str(cnt))
plt.plot(train1[0:,1],train1[0:,2],".",label="训练数据集1")
plt.plot(train2[0:,1],train2[0:,2],".",label="训练数据集2")
plt.plot(test1[0:,1],test1[0:,2],".",label="测试数据集1")
plt.plot(test2[0:,1],test2[0:,2],".",label="测试数据集2")
plt.legend()
plt.show()



# %%
