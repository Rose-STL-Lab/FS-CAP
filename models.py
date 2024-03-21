from torch import nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        if config['encoder_batchnorm']:
            self.fc = nn.Sequential(nn.Linear(in_dim, config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.BatchNorm1d(config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['d_model']))
        else:
            self.fc = nn.Sequential(nn.Linear(in_dim, config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['layer_width']),
                                    nn.ReLU(),
                                    nn.Linear(config['layer_width'], config['d_model']))

    def forward(self, x, scalar=None):
        if scalar != None:
            return self.fc(x * scalar)
        return self.fc(x)


class Predictor(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        if config['pred_dropout']:
            if config['pred_batchnorm']:
                self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1))
            else:
                self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(config['pred_dropout_p']),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1))
        else:
            if config['pred_batchnorm']:
                self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1))
            else:
                self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                        nn.ReLU(),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1))

    def forward(self, x):
        return self.fc(x)
