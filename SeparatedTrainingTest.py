import os, shutil, random


def missing_elements(L):
    start, end = 0, 999
    return sorted(set(range(start, end + 1)).difference(L))

# nombre de donnée
nbDonnee = 1000

# pourcentage de séléction training
pourcentage = 0.8

# séléction des données format tableau
choixNegTraining = random.sample(range(0000, (nbDonnee-1)), int(pourcentage*nbDonnee))
choixNegTest = missing_elements(choixNegTraining)
choixNegTraining.sort()
choixNegTest.sort()

choixPosTraining = random.sample(range(0000, (nbDonnee-1)), int(pourcentage*nbDonnee))
choixPosTest = missing_elements(choixNegTraining)
choixPosTraining.sort()
choixPosTest.sort()

#définition des catégories
categorie = ["neg", "pos"]

#les mots intéréssant des différents fichiers
separateur = ["VER", "NOM", "ADJ", "ADV"]


# tableau des chemins des fichiers (destination et source)
pathTab = ["./Donnee_initial/tagged/neg/", "./Donnee_initial/tagged/neg/", "./Donnee_initial/tagged/pos/",
           "./Donnee_initial/tagged/pos/"]

moveToTab = ["./TrainingSet/tagged/neg/", "./TestSet/tagged/neg/", "./TrainingSet/tagged/pos/", "./TestSet/tagged/pos/"]

prefixTab = ["neg-", "neg-", "pos-", "pos-"]

dataTab = [choixNegTraining, choixNegTest, choixPosTraining, choixPosTest]


# sépares les données initiales dans les testset et trainingset
for f in range(0, 4):
    for i in dataTab[f]:
        file = prefixTab[f] + str(i).zfill(4) + ".txt"
        src = pathTab[f] + file

        wordFind = ""

        # ouverture en lecture du fichier de base et parcours pour trouver les mots
        fileSource = open(src, 'r')
        for line in fileSource:
            for word in line.split("\t"):
                if word in separateur:

                    # récupere le dernier mot de la ligne de 3
                    tabWordFind = line.split("\t")[2]

                    # ajout le mot au string  qui sera écrit dans le fichier
                    wordFind += tabWordFind.split("|")[0] if "|" in tabWordFind else tabWordFind
        fileSource.close()

        # écriture dans le fichier
        dst = moveToTab[f] + file
        fileDestination = open(dst, 'w')
        fileDestination.write(wordFind)
        fileDestination.close()


