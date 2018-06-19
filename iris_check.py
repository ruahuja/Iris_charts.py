#!user/bin/python3

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

#loading iris datasets

iris=load_iris()

#trainig flowers features stored in iris.data
#output accordingly stored in iris.target

#now splitting into test and train sets

train_iris, test_iris, train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.1)

#calling decisiontree classifier
dsclf=tree.DecisionTreeClassifier()

#calling KNN algo
knnclf=KNeighborsClassifier(n_neighbors=5)

#training data
traineddsc=dsclf.fit(train_iris,train_target)
trainedknn=knnclf.fit(train_iris,train_target)

#testing algo
outputdsc=traineddsc.predict(test_iris)
print(outputdsc)
outputknn=trainedknn.predict(test_iris)
print(outputknn)

#orignal output
print(test_target)

#calculating accuracy decisiontree
pctd=accuracy_score(test_target,outputdsc)
print(pctd)

#calculating accuracy for knn
pct=accuracy_score(test_target,outputknn)
print(pct)


