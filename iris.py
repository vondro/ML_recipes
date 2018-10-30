import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# load the iris toy dataset from Scikit-learn
iris = load_iris()

# 2 samples will be taken out of training data to create testing data
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris['target'], test_idx)
train_data = np.delete(iris['data'], test_idx, axis=0)

# testing data
test_target = iris['target'][test_idx]
test_data = iris['data'][test_idx]

# create classifier and train it
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# let's compare test target to prediction on test data
print(test_target)
print(clf.predict(test_data))

# vizualization code from scikit tutorial
from sklearn.externals.six import StringIO
import graphviz
dot_data = StringIO()
tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True, rounded=True,
        impurity=False)
graph = graphviz.Source(dot_data.getvalue())
graph.render("iris.pdf", view=True)
