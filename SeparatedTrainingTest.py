import os, shutil, random

if __name__ == "__main__":

    def missing_elements(L):
        start, end = 0, nbDonnee - 1
        return sorted(set(range(start, end + 1)).difference(L))


    # nombre de donnee
    nbDonnee = 1000

    # pourcentage de selection training
    pourcentage = 0.8

    # selection des donnees format tableau
    choixNegTraining = random.sample(range(0000, (nbDonnee - 1)), int(pourcentage * nbDonnee))
    choixNegTest = missing_elements(choixNegTraining)
    choixNegTraining.sort()
    choixNegTest.sort()

    choixPosTraining = random.sample(range(0000, (nbDonnee - 1)), int(pourcentage * nbDonnee))
    choixPosTest = missing_elements(choixNegTraining)
    choixPosTraining.sort()
    choixPosTest.sort()

    # definition des categories
    categorie = ["neg", "pos"]

    # les mots interessant des differents fichiers
    separateur = ["VER", "NOM", "ADJ", "ADV"]

    # tableau des chemins des fichiers (destination et source)
    pathTab = ["./Donnee_initial/tagged/neg/", "./Donnee_initial/tagged/neg/", "./Donnee_initial/tagged/pos/",
               "./Donnee_initial/tagged/pos/"]

    moveToTab = ["./TrainingSet/tagged/neg/", "./TestSet/tagged/neg/", "./TrainingSet/tagged/pos/",
                 "./TestSet/tagged/pos/"]

    prefixTab = ["neg-", "neg-", "pos-", "pos-"]

    dataTab = [choixNegTraining, choixNegTest, choixPosTraining, choixPosTest]

    print("verifier que les chemins sont correcte pour les donnees source :")
    print(pathTab)

    print("verifier que les chemins sont correcte pour les donnees destination : \n")
    print(moveToTab)

    lastFile = ""
    print(
        " \n demarrage du parse de tout les documents ainsi que leurs ecritures dans le dossier TestSet et TrainingSet \n")
    # separes les donnees initiales dans les testset et trainingset
    for f in range(0, 4):
        for i in dataTab[f]:
            file = prefixTab[f] + str(i).zfill(4) + ".txt"
            src = pathTab[f] + file

            wordFind = ""

            # ouverture en lecture du fichier de base et parcours pour trouver les mots
            fileSource = open(src, 'r')
            for line in fileSource:
                for word in line.split("\t"):
                    if ':' in word:
                        tabwsep = word.split(":")
                        for w in tabwsep :
                            if w in separateur:
                                # recupere le dernier mot de la ligne de 3
                                tabWordFind = line.split("\t")[2]

                                # ajout le mot au string  qui sera ecrit dans le fichier
                                wordFind += tabWordFind.split("|")[0] if "|" in tabWordFind else tabWordFind
                    if word in separateur:
                        # recupere le dernier mot de la ligne de 3
                        tabWordFind = line.split("\t")[2]

                        # ajout le mot au string  qui sera ecrit dans le fichier
                        wordFind += tabWordFind.split("|")[0] if "|" in tabWordFind else tabWordFind
            fileSource.close()

            # ecriture dans le fichier
            dst = moveToTab[f] + file
            lastFile = dst
            fileDestination = open(dst, 'w')
            fileDestination.write(wordFind)
            fileDestination.close()

    print("fin de l ecriture dans les dossiers \n")
    print("test de lecture du dernier fichier :\n")
    print("file = " + str(lastFile) + "\n")

    fileFinal = open(lastFile, 'r')
    for line in fileFinal:
        print(line + "\t")

    print("fin des test de lectures. vous pouvez executer le fichier Classification.py")
