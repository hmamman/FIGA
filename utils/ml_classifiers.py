from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    'dt': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(n_estimators=100),
    'mlp': MLPClassifier(max_iter=1000),
    'svm': SVC(probability=True),
    'lr': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000)
}

    #           Bank    Credit  Census  Compas  Meps
    #   DT:     0.8714  0.7750  0.8196  0.8129  0.7907
    #   RF:     0.9025  0.7917  0.8535  0.8299  0.8737
    #   MLP:    0.8964  0.7167  0.8426  0.8284  0.8466
    #   SVM:    0.8844  0.7333  0.8142  0.8266  0.8622
    #   LR:     0.8956  0.7250  0.8131  0.8266  0.8797