import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fontMat
import numpy as np
import random

# 设置中文显示
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创造数据方式1


def getDataT(noiseScale, numScale):
    X = np.linspace(0, 1, numScale)
    Y = np.sin(2*np.pi*X)+np.random.normal(0, noiseScale,
                                           size=numScale)  # 添加零均值，方差为noiseScale的高斯噪声
    NumData = []
    for i in range(0, numScale):
        NumData.append([X[i], Y[i]])
    return NumData

# 创造数据方式2


def getData(noiseScale, numScale):
    X = np.linspace(0, 1, numScale)
    Y = np.sin(2*np.pi*X)+np.random.normal(0, noiseScale,
                                           size=numScale)  # 添加零均值，方差为noiseScale的高斯噪声
    return np.array(X).reshape(numScale, 1), np.array(Y).reshape(numScale, 1)

# 获取X矩阵


def getXMatrix(xData, nScale):
    XMatrix = []
    for i in range(0, len(xData)):
        XMatrix.append([(lambda j:pow(xData[i], (j-1)))(j)
                       for j in range(1, nScale+1)])
    return np.array(XMatrix)

# 解析解求解函数


def getParamWithoutPenalty(XMatrix, TMatrix):
    return np.linalg.inv(XMatrix.T@XMatrix)@XMatrix.T@TMatrix


def getParamWithPenalty(XMatrix, TMatrix, lambdaPenalty):
    return np.linalg.inv(XMatrix.T@XMatrix+lambdaPenalty*np.eye(XMatrix.shape[1], XMatrix.shape[1]))@XMatrix.T@TMatrix

# 误差函数


def calcLoss(xMatrix, wVec, tVec, lambdaPenalty):
    return 0.5*np.mean((xMatrix@wVec-tVec).T@(xMatrix@wVec-tVec)+lambdaPenalty*(wVec.T@wVec))

# 画曲线


def showSinPlot():
    xData = np.linspace(0, 1, 150)
    plt.plot(xData, np.sin(2*np.pi*xData), color='black', label="被拟合的正弦曲线")


def showTrainResult(xTrain, yTrain, lineColor, label):
    plt.scatter(xTrain, yTrain, color=lineColor, label=label)


def showPredResult(xData, yPred, lineColor, label):
    plt.plot(xData, yPred, color=lineColor, label=label)

# 梯度下降法


class GradientDescentOptimizer:
    def __init__(self, model, nScale, learningRate, lossLimit, lossFunction):
        self.model = model
        self.nScale = nScale
        self.learningRate = learningRate
        self.lossLimit = lossLimit
        self.lossFunction = lossFunction

    def calcGradient(self, xMatrix, wVec, tVec, lambdaPenalty):
        return xMatrix.T@xMatrix@wVec-xMatrix.T@tVec+lambdaPenalty*wVec

    def optimizeParam(self, xMatrix, wVec, tVec, lambdaPenalty):
        # 观测优化过程以及优化次数的限制
        ck = 0
        lossArray = []
        # 记录Loss值
        lastLoss = self.lossFunction(xMatrix, wVec, tVec, lambdaPenalty)
        loss = lastLoss
        while True:
            wVecTmp = wVec-self.learningRate * \
                self.calcGradient(xMatrix, wVec, tVec, lambdaPenalty)
            lossTmp = self.lossFunction(xMatrix, wVecTmp, tVec, lambdaPenalty)
            if lossTmp > lastLoss:
                self.learningRate *= 0.5
            else:
                wVec = wVecTmp
                loss = lossTmp
                if (np.abs(loss-lastLoss) < self.lossLimit) and (loss <= 1.3 or wVec.shape[0] <= 3):
                    break
                lastLoss = loss
                lossArray.append(loss)
            ck += 1
        plt.plot(np.array(lossArray))
        plt.show()
        return wVec, ck

# 共轭梯度法


class CGOptimizer:
    def __init__(self, deltaLimit):
        self.deltaLimit = deltaLimit

    def optimizeParam(self, xMatrix, wVec, tVec, lambdaPenalty):
        # Make Data as AX=b
        A = xMatrix.T@xMatrix+lambdaPenalty*np.identity(wVec.shape[0])
        X = wVec
        b = xMatrix.T@tVec

        RLast = b-A@X
        P = RLast
        ck = 0
        while ck < wVec.shape[0]:
            alpha = (P.T@RLast)/(P.T@A@P)
            R = RLast-(alpha*A)@P
            X = X+alpha*P
            if RLast.T@RLast < self.deltaLimit:
                break
            P = R+((R.T@R)/(RLast.T@RLast))*P
            RLast = R
            ck += 1

        return X, ck


def getBestLambda(xTrain, yTrain, xTest, yTest):
    lambdaTrainLoss = []
    lambdaTestLoss = []
    for j in range(-30, 0):
        # 训练参数
        N_SCALE = 12                # 多项式阶数
        LAMBDA_PENALTY = np.exp(j)  # 惩罚项系数
        XTrainMatrix = getXMatrix(xTrain, N_SCALE).reshape(
            TRAIN_NUM_SCALE, N_SCALE)
        XTestMatrix = getXMatrix(xTest, N_SCALE).reshape(TEST_SCALE, N_SCALE)
        # 解析解方法
        ParamWithPenalty = getParamWithPenalty(
            XTrainMatrix, yTrain, LAMBDA_PENALTY)
        lambdaTrainLoss.append([LAMBDA_PENALTY, calcLoss(
            XTrainMatrix, ParamWithPenalty, yTrain, LAMBDA_PENALTY)])
        lambdaTestLoss.append([LAMBDA_PENALTY, calcLoss(
            XTestMatrix, ParamWithPenalty, yTest, LAMBDA_PENALTY)])
    lambdaTrainLoss = np.array(lambdaTrainLoss)
    lambdaTestLoss = np.array(lambdaTestLoss)
    print(lambdaTrainLoss[0:, 1])
    plt.plot(np.log(lambdaTrainLoss[0:, 0]), lambdaTrainLoss[0:, 1])
    plt.plot(np.log(lambdaTrainLoss[0:, 0]), lambdaTestLoss[0:, 1])


if __name__ == '__main__':
    # 生成数据参数
    NOISE_SCALE = 0.3  # 噪声方差
    TRAIN_NUM_SCALE = 20  # 训练集规模
    TEST_SCALE = 15  # 测试集规模

    NumData = np.random.permutation(
        getDataT(NOISE_SCALE, TRAIN_NUM_SCALE+TEST_SCALE))
    TrainData = np.array(sorted(NumData[0:TRAIN_NUM_SCALE, ].reshape(
        TRAIN_NUM_SCALE, 2), key=lambda x: x[0]))
    TestData = np.array(
        sorted(NumData[TRAIN_NUM_SCALE:, ].reshape(TEST_SCALE, 2), key=lambda x: x[0]))
    xTrain = TrainData[0:, 0].reshape(TRAIN_NUM_SCALE, 1)
    yTrain = TrainData[0:, 1].reshape(TRAIN_NUM_SCALE, 1)
    xTest = TestData[0:, 0].reshape(TEST_SCALE, 1)
    yTest = TestData[0:, 1].reshape(TEST_SCALE, 1)

    for i in [2, 4, 8, 12]:
        # 训练参数
        N_SCALE = i  # 多项式阶数
        LAMBDA_PENALTY = 1e-7  # 惩罚项系数
        LEARNING_RATE = 0.01  # 学习率
        LOSS_LIMIT = 1e-6  # loss限制

        XTrainMatrix = getXMatrix(xTrain, N_SCALE).reshape(TRAIN_NUM_SCALE, N_SCALE)
        XTestMatrix = getXMatrix(xTest, N_SCALE).reshape(TEST_SCALE, N_SCALE)

        # 解析解方法
        ParamWithoutPenalty = getParamWithoutPenalty(XTrainMatrix, yTrain)
        ParamWithPenalty = getParamWithPenalty(XTrainMatrix, yTrain, LAMBDA_PENALTY)

        # 梯度下降
        wBGD = np.zeros(N_SCALE).reshape(N_SCALE, 1)
        optimizer = GradientDescentOptimizer('BGD', N_SCALE, LEARNING_RATE, LOSS_LIMIT, calcLoss)
        wBGD, ck = optimizer.optimizeParam(XTrainMatrix, wBGD, yTrain, LAMBDA_PENALTY)

        # 共轭梯度法
        wCG = np.zeros(N_SCALE).reshape(N_SCALE, 1)
        optimizer = CGOptimizer(LOSS_LIMIT)
        wCG, ck = optimizer.optimizeParam(XTrainMatrix, wCG, yTrain, LAMBDA_PENALTY)

        print("训练集loss=", calcLoss(XTrainMatrix, ParamWithoutPenalty, yTrain, 0))
        print("测试集loss=", calcLoss(XTestMatrix, ParamWithoutPenalty, yTest, 0))
        print("迭代次数：", ck)
        showSinPlot()
        showTrainResult(xTrain, yTrain, 'red', label="训练集点")
        showPredResult(xTrain, XTrainMatrix@ParamWithoutPenalty,'blue', label='不带惩罚项的解析解')
        showPredResult(xTrain, XTrainMatrix@ParamWithPenalty,'yellow', label="带惩罚项的解析解")
        showPredResult(xTrain, XTrainMatrix@wBGD, 'green', label="梯度下降法")
        showPredResult(xTrain, XTrainMatrix@wCG, 'purple', label="共轭梯度法")
        plt.title("多项式阶数为："+str(N_SCALE)+",训练集规模"+str(TRAIN_NUM_SCALE))
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
