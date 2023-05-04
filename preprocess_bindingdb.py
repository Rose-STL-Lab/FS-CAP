import csv
import math
import pickle
from tqdm import tqdm
import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles
from sklearn.model_selection import train_test_split
import collections
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import sys


def featurize_mol(smiles):
    return np.array(GetMorganFingerprintAsBitVect(MolFromSmiles(smiles), 3))


mols = []
f = open(sys.argv[1], 'r')
next(f)
assays = []
for i, row in tqdm(enumerate(csv.reader(f, delimiter='	'))):
    if (row[8] or row[9] or row[10] or row[11]) and (10 <= len([char for char in row[1] if char not in '()=@[]123456789']) <= 70) and row[37] != 'NULL' and MolFromSmiles(row[1]):
        val = (row[10] if row[10] else (row[8] if row[8] else (row[9] if row[9] else row[11]))).replace('<', '').replace('>', '').strip()
        assays.append(row[37].upper())
        mols.append((MolToSmiles(MolFromSmiles(row[1])), math.log10(float(val) + 1e-10)))
allowed_assays = [assay for assay, count in collections.Counter(assays).most_common() if count >= 10]

training_assays, testing_assays = train_test_split(allowed_assays, test_size=100)
training_assays = set(training_assays)
testing_assays = set(testing_assays)
train_mols, train_assays = zip(*[(mols[i], assays[i]) for i in range(len(mols)) if assays[i] in training_assays])
test_mols, test_assays = zip(*[(mols[i], assays[i]) for i in range(len(mols)) if assays[i] in testing_assays])

x_train = np.array([featurize_mol(smiles) for smiles, _ in train_mols], dtype=bool)
x_test = np.array([featurize_mol(smiles) for smiles, _ in test_mols], dtype=bool)
y_train = np.array([binding for _, binding in train_mols])
y_test = np.array([binding for _, binding in test_mols])
pickle.dump((x_train, x_test, y_train, y_test, train_assays, test_assays), open('bindingdb_data.pickle', 'wb')) 