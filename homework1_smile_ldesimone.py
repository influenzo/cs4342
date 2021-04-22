#Lorenzo DeSimone

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


#Takes in a vector of ground-truth labels and corresponding vector of guesses
#Computes the accuracy (PC) (one line)

def fPC (y, yhat):
    return np.mean(y == yhat)

#Takes in a set of predictors, a set of images to run it on, as well as the ground-truth labels of that set.
#For each image in the image set, it runs the ensemble to obtain a prediction
#Computes and returns the accuracy (PC) of the predictions w.r.t. the ground-truth labels.
def measureAccuracyOfPredictors (predictors, X, y):
 #Initialization at value of 0
    count = np.zeros(y.shape)
    for x in predictors:
        r1, c1, r2, c2 = x

        #Compare to determine if the current pixel at r1,c1 is brighter than the compared pixel at r2,c2
        diff = X[:,r1,c1] - X[:,r2,c2]
        #Designating if it is brighter or darker
        diff[diff > 0] = 1
        diff[diff <= 0] = 0
        #if brighter, increase the count
        count = count + diff
    #create a mean from the count and the predictors
    mean = count / len(predictors)

    #designate a given predication as true (>0.5) or false (<=0.5)
    mean[mean > 0.5] = 1
    mean[mean <= 0.5]= 0

    #Accuracy Calculating
    return fPC(y, mean)


def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):

    #initalize array of predictors and a variable for the best accuracy for that sample size
    predictors = []
    bestAccuracy = []

    for m in range(0,4):
        #initalizing variables for the best accuracy and best pair of pixels, to be updated as the machine iterates
        bestAccuracy = 0
        bestFeature = None
        #iterate through each of the 24 pixels via for-loops
        for r1 in range(0,23):
            for c1  in range(0,23):
                for r2  in range(0,23):
                    for c2  in range(0,23):
                        #eliminate the potential for duplicate comparisons
                        if (r1,c1) == (r2,c2):
                            continue
                        #"It’s theoretically possible for the best ensemble to consist of the same feature multiple times" - Professor Whitehill
                        nextPredictors = predictors + list(((r1,c1,r2,c2),))
                        testAcc = measureAccuracyOfPredictors(nextPredictors, trainingFaces, trainingLabels)

                        if testAcc > bestAccuracy:
                            bestAccuracy = testAcc
                            bestFeature = (r1,c1,r2,c2)

        predictors.append(bestFeature)

    r1,c1,r2,c2 = bestFeature

    #reporting results for a given sample size
    print('Pair of Coordinates: ', bestFeature, "\n")
    print('Training Accuracy ',bestAccuracy, "\n")
    print("Testing Accuracy", testAcc, "\n")

    return predictors

#visual representation of the machine, pretty much entirely from homework1_smile.py
def viz(predictors, testingFaces):
    #3 is my favorite number
    im = testingFaces[3,:,:]
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')

    for x in predictors:
        r1,c1,r2,c2 = x
        # Show r1,c1
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        plt.show()


#analyze Training and Testing accuracy changes as a function of number of examples n ∈ {400, 800, 1200, 1600, 2000}
def runnit (trainingFaces, trainingLabels, testingFaces, testingLabels):

    sSizes = [400, 800, 1200, 1600, 2000]

    preedictors = []
    accuracyStorage = []
    for x in sSizes:

        #First print the given sample size, then run the regression, printing the corresponding Training and Testing Accuracies
        print("Sample Size: ", x, "\n")
        preedictors = stepwiseRegression(trainingFaces[:x],trainingLabels[:x], testingFaces, testingLabels)

    viz(preedictors, testingFaces)

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    runnit(trainingFaces,trainingLabels, testingFaces, testingLabels)
