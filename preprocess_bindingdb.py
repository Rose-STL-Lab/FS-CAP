import csv
import math
import pickle
from tqdm import tqdm
import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles
from sklearn.model_selection import train_test_split
import collections
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


def featurize_mol(smiles):
    return np.array(GetMorganFingerprintAsBitVect(MolFromSmiles(smiles), 3))


mols = []
f = open('BindingDB_All.tsv', 'r')
next(f)
seqs = []
for i, row in tqdm(enumerate(csv.reader(f, delimiter='	'))):
    # 8 or 10 for ki/kd, 9 or 11 for ic50/ec50
    if (row[8] or row[9] or row[10] or row[11]) and (10 < len([char for char in row[1] if char not in '()=@[]123456789']) < 70) and row[37] != 'NULL' and MolFromSmiles(row[1]):
        val = (row[10] if row[10] else (row[8] if row[8] else (row[9] if row[9] else row[11]))).replace('<', '').replace('>', '').strip()
        seqs.append(row[37].upper())
        mols.append((MolToSmiles(MolFromSmiles(row[1])), math.log10(float(val) + 1e-10)))

allowed_seqs = [seq for seq, count in collections.Counter(seqs).most_common() if count > 10]

for seq in tqdm(allowed_seqs):
    vals = [mols[i][1] for i in range(len(mols)) if seqs[i] == seq]
    if not (True in [4 < val < 50 for val in vals]):
        i = 0
        while i < len(mols):
            if seqs[i] == seq:
                del mols[i]
                del seqs[i]
            else:
                i += 1
allowed_seqs = [seq for seq, count in collections.Counter(seqs).most_common() if count > 10]

training_seqs, testing_seqs = train_test_split(allowed_seqs, test_size=100)
training_seqs = set(training_seqs)
testing_seqs = set(testing_seqs)
train_mols, train_seqs = zip(*[(mols[i], seqs[i]) for i in range(len(mols)) if seqs[i] in training_seqs])
test_mols, test_seqs = zip(*[(mols[i], seqs[i]) for i in range(len(mols)) if seqs[i] in testing_seqs])
y_train = np.array([binding for _, binding in train_mols])
y_test = np.array([binding for _, binding in test_mols])

x_train = np.array([featurize_mol(smiles) for smiles, _ in train_mols], dtype=bool)
x_test = np.array([featurize_mol(smiles) for smiles, _ in test_mols], dtype=bool)
pickle.dump((x_train, x_test, y_train, y_test, train_seqs, test_seqs), open('bindingdb_data.pickle', 'wb'))