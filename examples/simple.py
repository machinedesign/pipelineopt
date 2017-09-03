from pipelineopt.estimator import Classifier

from sklearn.datasets import load_digits

data = load_digits()
X = data['images']
y = data['target']
X = X.reshape((X.shape[0], -1))
clf = Classifier(nb_iter=10)
clf.fit(X, y)
print('Best code :')
print(clf.best_estimator_code_)
print('Best score : {}'.format(clf.best_score_))
