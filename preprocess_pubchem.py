import math
import pickle
from tqdm import tqdm
import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles
from sklearn.model_selection import train_test_split
import collections
import sys
import os
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


def featurize_mol(smiles):
    return np.array(GetMorganFingerprintAsBitVect(MolFromSmiles(smiles), 3))


sid_to_smiles = {}
for f_name in os.listdir(sys.argv[2]):
    if f_name.endswith('.smi'):
        for line in open(f'{sys.argv[2]}/{f_name}', 'r'):
            try:
                smiles, sid = line.strip().split()
                sid_to_smiles[sid] = smiles
            except:
                pass
mols = []
assays = []
for line in tqdm(open(sys.argv[1], 'r')):
    aid, sid, val = line.strip().split()
    try:
        smiles = sid_to_smiles[sid]
    except:
        continue
    if MolFromSmiles(smiles) and (10 <= len([char for char in smiles if char not in '()=@[]123456789']) <= 70):
        assays.append(aid)
        mols.append((MolToSmiles(MolFromSmiles(smiles)), math.log10(float(val) * 1000 + 1e-10)))
allowed_assays = [assay for assay, count in collections.Counter(assays).most_common() if count >= 10]

training_assays, testing_assays = train_test_split(allowed_assays, test_size=1000)
training_assays = set(training_assays)
testing_assays = set(testing_assays)
train_mols, train_assays = zip(*[(mols[i], assays[i]) for i in range(len(mols)) if assays[i] in training_assays])
test_mols, test_assays = zip(*[(mols[i], assays[i]) for i in range(len(mols)) if assays[i] in testing_assays])

x_train = np.array([featurize_mol(smiles) for smiles, _ in train_mols], dtype=bool)
x_test = np.array([featurize_mol(smiles) for smiles, _ in test_mols], dtype=bool)
y_train = np.array([binding for _, binding in train_mols])
y_test = np.array([binding for _, binding in test_mols])
pickle.dump((x_train, x_test, y_train, y_test, train_assays, test_assays), open('pubchem_data.pickle', 'wb'))