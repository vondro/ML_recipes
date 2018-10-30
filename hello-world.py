from sklearn import tree

# features
# oranges - heavier, bumpy=0
# apples - lighter, smooth=1

# features = [[140, 'smooth'], [130, 'smooth'], [150, 'bumpy'], [170, 'bumpy']]
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# labels
# apple=0, orange=1
labels = [0, 0, 1, 1]

# train the classifier - we will use decision tree
# classifier = box of rules
clf = tree.DecisionTreeClassifier()

# fit - synonym to "find patterns in data"
clf = clf.fit(features, labels)

# this should be an orange
print(clf.predict([[150, 0]]))
