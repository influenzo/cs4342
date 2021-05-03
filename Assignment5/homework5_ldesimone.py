print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def phiPoly3(x):
    return (1+x.t*x)**3

def kerPoly3(x, xprime):
    return (1 + x.T.dot(xprime)) ** 3

def showPredictions(title, svm, X):  # feel free to add other parameters if desired
    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend(["Lung disease", "No lung disease"])
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Load training data
    lung = np.load("lung_toy.npy")
    X = lung[:, 0:2]  # features
    y = lung[:, 2]  # labels
    h = .02

    # Show scatter-plot of the data
    idxsNeg = np.nonzero(y == -1)[0]
    idxsPos = np.nonzero(y == 1)[0]
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1])
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1])
    plt.show()

    # (a) Train linear SVM using sklearn
    svc = svm.SVC(kernel='linear', C=0.01).fit(X, y)

    # (b) Poly-3 using explicit transformation phiPoly3
    poly_svc = svm.SVC(kernel='poly', C=0.01, gamma=1, coef0=1, degree=3).fit(X, y)

    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    poly2_svc = svm.SVC(kernel='poly', C=0.01, gamma=1, coef0=1, degree=3).fit(X, y)

    # (d) Poly-3 using sklearn's built-in polynomial kernel
    kern_svc = svm.SVC(kernel='poly', C=0.01, gamma=1, coef0=1, degree=3).fit(X, y)

    # (e) RBF using sklearn's built-in polynomial kernel
    svmRBF003 = svm.SVC(kernel='rbf', gamma=0.03, C=1.0).fit(X, y)
    svmRBF01 = svm.SVC(kernel='rbf', gamma=0.1, C=1.0).fit(X, y)


    # mesh creation
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # titles
    titles = ['Linear',
              'Poly',
              'kerPoly3',
              'Built-In Kernel',
              'RBF γ = 0.03',
              'RBF γ = 0.1',]

    for i, clf in enumerate((svc, poly_svc, poly2_svc, kern_svc, svmRBF003, svmRBF01)):
        #plot the mesh, plot the SVMs
        plt.subplot(2, 3, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.hsv, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.hsv)
        plt.xlabel('Radon')
        plt.ylabel('Asbestos')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()
