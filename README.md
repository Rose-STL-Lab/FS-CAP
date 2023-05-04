# Few-Shot Compound Activity Prediction (FS-CAP)

This repository contains code for the few-shot compound activity prediction (FS-CAP) algorithm (see our paper at ). Unfortunately, no pretrained models could be provided due to upload size constraints.

## Requirements
[RDKit](https://www.rdkit.org/docs/Install.html) is required, and [OpenBabel](https://openbabel.org/docs/dev/Installation/install.html) if one wishes to download PubChemBA. All code was tested in Python 3.10. The following pip packages are also required:
```
torch
scipy
scikit-learn
numpy
tqdm
requests (to download PubChemBA)
```

## Preprocessing
We only provide code to preprocess the datasets that were used for training, but testing on other datasets using a trained model should be relatively straightforward using the `score_compounds.py` script.

### BindingDB
`preprocess_bindingdb.py` contains code to extract and preprocess data from BindingDB. Calling `python preprocess_bindingdb.py <filename>` will load data from `<filename>` in the BindingDB tsv format, and produce a `bindingdb_data.pickle` file that is ready for training. For the paper, we used `BindingDB_All.tsv` from BindingDB's [Download](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?all_download=yes) page, available [here](https://www.bindingdb.org/bind/downloads/BindingDB_All_2022m8.tsv.zip). 

### PubChemBA
Downloading data from PubChem is more involved than BindingDB, and involves the script `download_pubchem.py`. This script will search PubChem's FTP server and download each assay datafile one-by-one, and will also download substance files as SDFs to later link the substance IDs (SIDs) in the assay files with SMILES. The script will print out each compound activity line-by-line to stdout for later preprocessing, so it should be run using `python download_pubchem.py <folder> > pubchem_data.txt`, where `<folder>` is the folder where the SDFs should be downloaded to. Then, calling `python preprocess_pubchem.py pubchem_data.txt <folder>` will preprocess this data and save a file `pubchem_data.pickle` that is ready for training.

## Training
`train.py` contains the main script to train FS-CAP. `--num_context_compounds` specifies the number of context compounds used for training and testing (default 8), `--dataset_name` specifies the prefix of the data file (e.g. `bindingdb` to train on `bindingdb_data.pickle`; default `bindingdb`), `--run_name` optionally specifies a name for TensorBoard logging, and `--checkpoint_file` will save model checkpoints to the provided filename (default `model.pt`). TensorBoard logs from training the model are saved to the `./runs/` folder, and includes both training and testing statistics. Other model hyperparameters can be found and adjusted in the `config` variable in the `train.py` file.

## Inference
`score_compounds.py` uses the trained model to perform inference on a given set of context and query compounds. The following parameters must be supplied: `--context_smiles` specifies the SMILES strings of the context molecules, separated by semicolons (e.g. `CCC;CCCC;CCCCC`), and `--context_activities` specifies the associated activites in log10 nanomoles/liter (nM) (e.g. `3;0;1` if the activites are 1000 nM, 1 nM, and 10 nM, respectively). `--query_smiles` specifies the SMILES string(s) of the query molecule(s) (if multiple, separate with semicolons), `--model_file` specifies the path to the trained model (default `model.pt`), and `encoding_dim` specifies the `encoding_dim` parameter used in training (default 512). The script prints to stdout the activity prediction of the query molecule(s) in nM, one prediction per line.