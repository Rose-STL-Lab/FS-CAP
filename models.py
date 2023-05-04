from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, config['encoding_dim']))

    def forward(self, x, scalar=None):
        if scalar != None:
            return self.fc(x * scalar)
        return self.fc(x)


class Predictor(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 2048),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.BatchNorm1d(2048),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.BatchNorm1d(2048),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.BatchNorm1d(2048),
                                nn.Linear(2048, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1))

    def forward(self, x):
        return self.fc(x)
