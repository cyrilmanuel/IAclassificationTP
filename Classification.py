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

if __name__ == "__main__":

    print("----------------------------------------------------\n")
    print("Demarrage du tp.\n ")
    print("----------------------------------------------------\n")
    print("définition des catégories : \n")

    # Determine les catégories
    categories = ['neg', 'pos']
    print(str(categories) + "\n")

    print("----------------------------------------------------\n")
    print("chargement des données de Training et Test \n")
    print("----------------------------------------------------\n")
    # chargement des données training et test
    set_Training = sklearn.datasets.load_files(
        "./TrainingSet/tagged/",
        description="Training data Tagged.",
        categories=categories, load_content=True, shuffle=True, encoding='latin-1', decode_error='strict',
        random_state=42, )

    print("longueur des données training chargées : "+str(len(set_Training.data))+"\n")
    set_Test = sklearn.datasets.load_files(
        "./TestSet/tagged/",
        description="Test data Tagged.",
        categories=categories, load_content=True, shuffle=True, encoding='latin-1', decode_error='strict',
        random_state=42, )

    print("longueur des données Test chargées : " + str(len(set_Test.data)) + "\n")

    # -----------------------
    # -- 2. vectorisation  --
    # -----------------------
    print("----------------------------------------------------\n")
    print("vectorisation des données\n")
    print("----------------------------------------------------\n")
    vectCount = CountVectorizer()
    Xtraining = vectCount.fit_transform(set_Training.data)
    print("nombre de fichier et de mots"+str(Xtraining.shape)+"\n")

    # -----------------------
    # -- 3. indexiations  ---
    # -----------------------
    print("----------------------------------------------------\n")
    print("indexiations des données\n")
    print("----------------------------------------------------\n")
    tfidf_Transfo = TfidfTransformer()  # Transform a count matrix to a normalized tf or tf-idf representation
    XtraininigTfidf = tfidf_Transfo.fit_transform(Xtraining)
    print("nombre de fichier et de mots" + str(XtraininigTfidf.shape) + "\n")

    # ------------------------------------
    # -- 4. classification naive bayes  --
    # ------------------------------------
    print("----------------------------------------------------\n")
    print("classification naive de bayes \n")
    print("----------------------------------------------------\n")
    clf = MultinomialNB().fit(XtraininigTfidf, set_Training.target)

    print("première prédiction sans pipeline et sur des mots \n")
    print("----------------------------------------------------\n")

    docs_new = ['bien', 'mauvais']
    print("mots à évaluer (prédire) : \n"+str(docs_new)+"\n")
    X_new_counts = vectCount.transform(docs_new)
    X_new_tfidf = tfidf_Transfo.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    print("resultat de la prédiction sur les mots \n")
    for doc, categories in zip(docs_new, predicted):
        print('le mot : %r => %s' % (doc, set_Training.target_names[categories]))


    # ---------------------
    # -- 5. Evaluations  --
    # ---------------------

    print("----------------------------------------------------\n")
    print("Création d'un pipeline Bayes et SVM avec le train Test \n")
    print("----------------------------------------------------\n")

    docs_test = set_Test.data  # Test documents

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

    print("\tPrécision des prediction BAYES \t : {0}".format(np.mean(predicted_NB == set_Test.target)))
    print("\tPrécision des prediction SVM \t\t : {0}".format(np.mean(predicted_SGDC == set_Test.target)))

    # ---------------------------------
    # -- 6. utilisation des metrics  --
    # ---------------------------------

    print("----------------------------------------------------\n")
    print("utilisation de l'analyse de resultat avec metrics \n")
    print("----------------------------------------------------\n")

    print("\n\n Metrics")

    print("\n\nClassification Bayes")
    print(metrics.classification_report(set_Test.target, predicted_NB, target_names=set_Test.target_names))
    print("Confusion matrix Bayes")
    print(metrics.confusion_matrix(set_Test.target, predicted_NB))

    print("\n\nClassification SVM")
    print(metrics.classification_report(set_Test.target, predicted_SGDC, target_names=set_Test.target_names))
    print("Confusion matrix SVM")
    print(metrics.confusion_matrix(set_Test.target, predicted_SGDC))

    # -----------------------
    # -- 7. Optimisations  --
    # -----------------------

    print("----------------------------------------------------\n")
    print("optimisations et tuning avec la grille de recherche \n")
    print("----------------------------------------------------\n")

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }

    gs_clf = GridSearchCV(text_clf_SGDC, parameters, n_jobs=-1)

    # Try fit on a subset of data
    gs_clf = gs_clf.fit(set_Training.data[:400], set_Training.target[:400])

    demoString = 'parfait jolie magnifique superbe '

    print("\nla prediction pour \"{0}\" : {1}".format(demoString, set_Training.target_names[
        gs_clf.predict([demoString])[0]]))

    print("\nle meilleure score avec SVM {0}".format(gs_clf.best_score_))

    print("\n parametres choisis:\n")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    print("----------------------------------------------------\n")
    print("-------------------- Fin du TP ---------------------\n")
    print("----------------------------------------------------\n")