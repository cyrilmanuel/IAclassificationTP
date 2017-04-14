import os, shutil, random


def missing_elements(L):
    start, end = 0, 999
    return sorted(set(range(start, end + 1)).difference(L))


# séléction des données format tableau
choixNegTraining = random.sample(range(0000, 999), 800)
choixNegTest = missing_elements(choixNegTraining)
choixNegTraining.sort()
choixNegTest.sort()

choixPosTraining = random.sample(range(0000, 999), 800)
choixPosTest = missing_elements(choixNegTraining)
choixPosTraining.sort()
choixPosTest.sort()

# tableau des chemins
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
        dst = moveToTab[f] + file
        shutil.copy(src, dst)


