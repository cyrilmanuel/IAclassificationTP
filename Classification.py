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


# 6. utilisation des metrics

print("\n\n Metrics")

print("\n\nClassification Bayes")
print(metrics.classification_report(set_test.target, predicted_NB, target_names=set_test.target_names))
print("Confusion matrix Bayes")
print(metrics.confusion_matrix(set_test.target, predicted_NB))

print("\n\nClassification SVM")
print(metrics.classification_report(set_test.target, predicted_SGDC, target_names=set_test.target_names))
print("Confusion matrix SVM")
print(metrics.confusion_matrix(set_test.target, predicted_SGDC))
