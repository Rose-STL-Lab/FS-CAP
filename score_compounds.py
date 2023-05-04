import argparse
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import torch
from models import *
from data import *
import math



class FSCAP:
    def __init__(self, model_file):
        self.context_encoder = Encoder(2048, config).cuda()
        self.query_encoder = Encoder(2048, config).cuda()
        self.predictor = Predictor(config['encoding_dim'] * 2, config).cuda()
        context_encoder_dict, query_encoder_dict, predictor_dict = torch.load(model_file)
        self.context_encoder.load_state_dict(context_encoder_dict)
        self.query_encoder.load_state_dict(query_encoder_dict)
        self.predictor.load_state_dict(predictor_dict)
        self.context_encoder.eval()
        self.query_encoder.eval()
        self.predictor.eval()

    def predict(self, context_smiles, context_activities, queries):
        context_x = torch.tensor(np.array([self.featurize_mol(smile) for smile in context_smiles], dtype=bool)).unsqueeze(0)
        context_y = torch.tensor(np.array([self.clip_activity(math.log10(float(activity) + 1e-10)) for activity in context_activities])).squeeze().unsqueeze(0)
        query_x = torch.tensor(np.array([self.featurize_mol(smile) for smile in queries], dtype=bool))
        context_x, context_y, query_x = context_x.to(dtype=torch.float32, device='cuda'), context_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1), query_x.to(dtype=torch.float32, device='cuda')
        context = torch.zeros((len(context_smiles), len(context_x), config['encoding_dim']), device='cuda')
        for j in range(len(context_smiles)):
            context[j] = self.context_encoder(context_x[:, j, :], context_y[:, j, :])
        context = context.mean(0)
        query = self.query_encoder(query_x)
        tiled_contexts = torch.zeros((len(queries), config['encoding_dim']), device='cuda')
        for i in range(len(queries)):
            tiled_contexts[i] = context
        x = torch.concat((tiled_contexts, query), dim=1)
        out = self.predictor(x)
        return (10 ** out.detach().cpu().flatten()).tolist()

    def featurize_mol(self, smiles):
        if not ((10 <= len([char for char in smiles if char not in '()=@[]123456789']) <= 70) and MolFromSmiles(smiles)):
            raise ValueError('smiles invalid or incorrect length')
        return np.array(GetMorganFingerprintAsBitVect(MolFromSmiles(smiles), 3))

    def clip_activity(self, val):
        if val < -2.5:
            val = -2.5
        if val > 6.5:
            val = 6.5
        return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_smiles', type=str)
    parser.add_argument('--context_activities', type=str)
    parser.add_argument('--query_smiles', type=str)
    parser.add_argument('--model_file', type=str, default='model.pt')
    parser.add_argument('--encoding_dim', type=int, default=512)
    args = parser.parse_args()
    config = {'encoding_dim': args.encoding_dim}
    fscap = FSCAP(args.model_file)
    for prediction in fscap.predict(args.context_smiles.split(';'), args.context_activities.split(';'), args.query_smiles.split(';')):
        print(prediction)