# pipelineopt

Pipelineopt is a sckit-learn pipeline (http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) 
optimizer. It provides a scikit-learn like `Classifier` class which implements `fit`, `predict`, and `predict_proba`. 
Within `fit`, several pipelines are tried and the best pipeline according to the validation score is selected.
The way pipelines are generated is using a context-free grammar defined `classifier_grammar`. Different ways
of generating from the grammars are called `Walkers`. The default `Walker` used is `RandomWalker` which selects
uniformly from the pipelines of the grammar. Trained/Trainable `Walkers` can also be used. For instance, an `RnnWalker`
can be initially pre-trained and use. See https://github.com/machinedesign/grammaropt for more details about `RnnWalker`.

# Example


''python
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
``
