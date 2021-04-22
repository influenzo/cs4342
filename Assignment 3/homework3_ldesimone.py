import math
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import *

def append1s(X):
    N, _ = X.shape
    return np.hstack((X, np.ones((N,1))))

def fPC(W, Xtilde, Y):
    Yhat = softmax(W, Xtilde)
    indicies1 = np.argmax(Yhat, axis=1)
    indicies2 = np.argmax(Y, axis=1)
    return np.sum(1*(indicies1 == indicies2)) / len(Y)

def fCE(W, Xtilde, Y):
    Yhat = softmax(W, Xtilde)
    sum = 0
    for i in range(10):
        sum += Y[:,i].dot(np.log(Yhat[:,i]))
    return -sum / len(Y)

def gradfCE(W, Xtilde, Y):
    Yhat = softmax(W, Xtilde)
    return Xtilde.T.dot(Yhat-Y) / len(Y)


def softmax(W, Xtilde):
    Z = Xtilde.dot(W).T
    Z = Z-Z.max(axis = 1, keepdims = True)
    Zexp = np.exp(Z)
    sumZexp = np.sum(Zexp, axis=0)
    return (Zexp / sumZexp[None,:]).T

def SGD(Xtilde, Y):
    n = len(Y)
    ntilde = 100
    EPOCH = 100
    ROUNDS = math.ceil(n/ntilde)
    epsilon = 1
    W = 0.01 * np.random.randn(Xtilde.shape[1], 10)
    for e in range(EPOCH):
        shuffledRounds = np.arange(ROUNDS)
        np.random.shuffle(shuffledRounds)
        for r in shuffledRounds:
            if r % 10 == 0:
                epsilon *= 0.98
            if r != 0:
                start = 100*r

                end = 100*(r+1)
                gradient = gradfCE(W, Xtilde[start:end, :], Y[start:end, :])
                W = W - epsilon * gradient
        if EPOCH - e <= 20:
            print("Batch", e, ": fCE:", fCE(W, Xtilde, Y), ",  fPC:", fPC(W, Xtilde, Y))
    return W
#Load data and make a copy to preserve the original data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_orig = test[:]

#make a BIG OL MATRIX
seperator = train.shape[0]
frames = [train, test]
titanic = pd.concat(frames)

# Sex is categorical (really holding back on making jokes here)
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic["Sex"] = titanic["Sex"].astype(int)

#Predictors that we're gonna use
predictors = ["Pclass", "Sex", "SibSp"]
# Pclass and SibSp are "okay" because they're numerical and therefore usable as-is


#7-10 split the previously combined matrix of the data set into test and train data
target = titanic["Survived"].iloc[:seperator]
train = titanic[predictors][:seperator]
test = titanic[predictors][seperator:]

#Create classifiers because we're working with categorical data
xgb = xgb.XGBClassifier(learning_rate = 0.05, n_estimators=500, use_label_encoder=False);
svmc = svm.SVC(C = 5, probability = True)

#fit that data
xgb.fit(train, target)
svmc.fit(train, target)
xgb_preds = xgb.predict_proba(test).transpose()[1]
svmc_preds = svmc.predict_proba(test).transpose()[1]

#Assign weights to the classifiers
ensemble_preds = xgb_preds*0.75 + svmc_preds*0.25

#determine definitive survival metrics
for x in range(len(ensemble_preds)):
    if ensemble_preds[x] >= 0.5:
        ensemble_preds[x] = 1
    else:
        ensemble_preds[x] = 0
results  = ensemble_preds.astype(int)

def oneHotVectors(labels):
    y = np.zeros((labels.size, labels.max()+1))
    y[np.arange(labels.size), labels] = 1
    return y

def visualize(vector):
    im = vector.reshape(28,28)
    _, ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    plt.show()

if __name__ == "__main__":
    fashion = False
    titanic = False

    if fashion:
        # Load data
        Xtr = np.load("fashion_mnist_train_images.npy")/255
        Ytr = np.load("fashion_mnist_train_labels.npy")
        Xte = np.load("fashion_mnist_test_images.npy")/255
        Yte = np.load("fashion_mnist_test_labels.npy")
        Xtilde_tr = append1s(Xtr)
        Xtilde_te = append1s(Xte)

        trainingVectors = oneHotVectors(Ytr)
        testingVectors = oneHotVectors(Yte)

        W = SGD(Xtilde_tr, trainingVectors)
        print("PC Accuracy for Training Set:", fPC(W, Xtilde_tr, Ytr))
        print("PC Accuracy for Testing Set:", fPC(W, Xtilde_te, Yte))
        print("CE Loss for Training Set:", fCE(W, Xtilde_tr, Ytr))
        print("CE Loss for Testing Set:", fCE(W, Xtilde_te, Yte))

    if titanic:
        # Load training data
        d = pd.read_csv("train.csv")
        y = d.Survived.to_numpy()
        sex = d.Sex.map({"male": 0, "female": 1}).to_numpy()
        Pclass = d.Pclass.to_numpy()

        # Compute predictions on test set
        svmc_preds = svmc.predict_proba(test).T[1]
        xgb_preds = xgb.predict_proba(test).T[1]
        W = softmax(Xtilde_tr, trainingVectors)


        # Assign different weightages to the classifiers
        ensemble_preds = xgb_preds * 0.75 + svmc_preds * 0.25

        for x in range(len(ensemble_preds)):
            if ensemble_preds[x] >= 0.5:
                ensemble_preds[x] = 1
            else:
                ensemble_preds[x] = 0

        results = ensemble_preds.astype(int)

        # Write CSV file of the format:
        submission = pd.DataFrame({"PassengerId": test_orig["PassengerId"], "Survived": results})
        submission.to_csv("homework3_titanic_ldesimone.csv", index=False)