# Few-Shot Compound Activity Prediction (FS-CAP)

This repository contains code for the few-shot compound activity prediction (FS-CAP) algorithm. 

## Docker instructions
First add a trained model file to the same folder as the `Dockerfile` (or use the one provided [here](https://drive.google.com/file/d/1SD8H5j6U7gyZOI_oncZrEzDBrz-z7-Ng/view?usp=sharing)), then run `sudo docker build -t fscap .` to build the container, then run `sudo docker run fscap` along with any command line arguments. For example, `sudo docker run fscap --context_smiles "<smiles>" --context_activities <activities> --query_smiles "<smiles>"`.

## Requirements
[RDKit](https://www.rdkit.org/docs/Install.html) is required. All code was tested in Python 3.10. The following pip packages are also required:
```
torch
scipy
scikit-learn
numpy
tqdm
```

## Preprocessing
We only provide code to preprocess BindingDB for training, but testing on other datasets using a trained model should be relatively straightforward using the `score_compounds.py` script.

### BindingDB
`preprocess_bindingdb.py` contains code to extract and preprocess data from BindingDB. Calling `python preprocess_bindingdb.py` will load data from `BindingDB_All.tsv` which should be placed in the folder beforehand, and after running it will produce a `bindingdb_data.pickle` file that is ready for training. For the paper, we used `BindingDB_All.tsv` from BindingDB's [Download](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?all_download=yes) page, available [here](https://www.bindingdb.org/bind/downloads/BindingDB_All_2022m8.tsv.zip). 

## Training
`train.py` contains the main script to train FS-CAP. By default, the model will train with 8 context compound and will save tensorboard logs to the `logs` folder. After training, it will save the model file to `model.pt`. Other model hyperparameters can be found and adjusted in the `config` variable in the `train.py` file.

## Inference
`score_compounds.py` uses the trained model to perform inference on a given set of context and query compounds. The following parameters must be supplied: `--context_smiles` specifies the SMILES strings of the context molecules, separated by semicolons (e.g. `CCC;CCCC;CCCCC`), and `--context_activities` specifies the associated activites in log10 nanomoles/liter (nM) (e.g. `3;0;1` if the activites are 1000 nM, 1 nM, and 10 nM, respectively). `--query_smiles` specifies the SMILES string(s) of the query molecule(s) (if multiple, separate with semicolons), `--model_file` specifies the path to the trained model (default `model.pt`), and `encoding_dim` specifies the `encoding_dim` parameter used in training (default 512). The script prints to stdout the activity prediction of the query molecule(s) in nM, one prediction per line.
