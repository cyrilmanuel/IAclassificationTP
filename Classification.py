import sklearn
import sklearn.datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


# 1. chargement des sets de données

# Determine les catégories
categories = ['neg', 'pos']

# chargement des données training et test
set_Training = sklearn.datasets.load_files(
    "./TrainingSet/tagged/",
    description="Training data Tagged.",
    categories=categories, load_content=True, shuffle=True, encoding='latin-1', decode_error='strict',
    random_state=42, )

set_test = sklearn.datasets.load_files(
    "./TestSet/tagged/",
    description="Test data Tagged.",
    categories=categories, load_content=True, shuffle=True, encoding='latin-1', decode_error='strict',
    random_state=42, )



# 2. vectorisation

vectCount = CountVectorizer()
Xtraining = vectCount.fit_transform(set_Training.data)

# 3. indexiation

tfidf_Transfo = TfidfTransformer()  # Transform a count matrix to a normalized tf or tf-idf representation
XtraininigTfidf = tfidf_Transfo.fit_transform(Xtraining)

# 4. Classification naive bayes
clf = MultinomialNB().fit(XtraininigTfidf, set_Training.target)

# 5. Évaluation

docs_test = set_test.data  # Test documents

# pipeline MultinomialNB (naïve Bayes)
text_clf_NB = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ])

# pipeline support vector machine (SVM)
text_clf_SGDC = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                          ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)), ])

text_clf_NB = text_clf_NB.fit(set_Training.data, set_Training.target)
_ = text_clf_SGDC.fit(set_Training.data, set_Training.target)

# Predictions
predicted_NB = text_clf_NB.predict(docs_test)
predicted_SGDC = text_clf_SGDC.predict(docs_test)

print("\tPrécision des prediction BAYES \t : {0}".format(np.mean(predicted_NB == set_test.target)))
print("\tPrécision des prediction SVM \t\t : {0}".format(np.mean(predicted_SGDC == set_test.target)))


############################################################################
################### STEP + | Tuning / Optimization #########################
############################################################################

'''
Classifiers tend to have many parameters. It is possible to run exhaustive search
of the best parameters on a grid of possible alues.
'''

# Possible parameters
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }

gs_clf = GridSearchCV(text_clf_SGDC, parameters, n_jobs=-1)  # Use all cores

# Try fit on a subset of data
gs_clf = gs_clf.fit(set_Training.data[:400], set_Training.target[:400])

demoString = 'Superbe génial aimer beau'

print("\nThe demo prediction for \"{0}\" is : {1}".format(demoString,
                                                          set_Training.target_names[gs_clf.predict([demoString])[0]]))

print("\n Meilleur score avec le SVM : {0}".format(gs_clf.best_score_))
print("\nFound with the following parameters :\n")
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, gs_clf.best_params_[param_name]))


