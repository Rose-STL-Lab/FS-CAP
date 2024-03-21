import torch
from torch import optim
from scipy.stats import linregress
import numpy as np
from models import *
from data import *
from torch.utils.tensorboard import SummaryWriter


context_num = 8


config = {
    'run_name': f'test',
    'context_ranges': [(-50, 50)] * context_num,  # unit is log10 nM
    'val_freq': 1024,
    'lr': 0.000040012,
    'layer_width': 2048,
    'batch_size': 1024,
    'warmup_steps': 128,
    'total_epochs': 2 ** 15,
    'n_heads': 16,
    'n_layers': 4,
    'affinity_embed_layers': 1,
    'init_range': 0.2,
    'scalar_dropout': 0.15766,
    'embed_dropout': 0.16668,
    'final_dropout': 0.10161,
    'pred_dropout': True,
    'pred_batchnorm': False,
    'pred_dropout_p': 0.1,
    'encoder_batchnorm': True
}

if config['simple']:
    config['dataloader_batch'] = 1024 // len(config['context_ranges'])
else:
    config['dataloader_batch'] = 128 // len(config['context_ranges'])


train_dataloader, test_dataloader = get_dataloaders(config['dataloader_batch'], config['context_ranges'])
context_encoder = MLPEncoder(2048, config).cuda()
query_encoder = MLPEncoder(2048, config).cuda()
predictor = Predictor(config['d_model'] * 2, config).cuda()

optimizer = optim.RAdam(list(context_encoder.parameters()) + list(query_encoder.parameters()) + list(predictor.parameters()), lr=config['lr'])

warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, 0.0001, 1, total_iters=config['warmup_steps'])
annealing_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['total_epochs'])
scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, annealing_scheduler], milestones=[config['warmup_steps']])
writer = SummaryWriter('logs/' + config['run_name'])

epoch = 0
while True:
    total_loss = 0
    count = 0
    for i, (context_x, context_y, query_x, query_y, _) in enumerate(train_dataloader):
        context_x, context_y, query_x, query_y = context_x.to(dtype=torch.float32, device='cuda'), context_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1), query_x.to(dtype=torch.float32, device='cuda'), query_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1)
        context = torch.zeros((len(config['context_ranges']), len(context_x), config['d_model']), device='cuda')
        for j in range(len(config['context_ranges'])):
            context[j] = context_encoder(context_x[:, j, :], context_y[:, j, :])
        context = context.mean(0)
        query = query_encoder(query_x)
        x = torch.concat((context, query), dim=1)
        loss = torch.mean((predictor(x) - query_y) ** 2)
        total_loss += loss.item()
        count += 1
        loss.backward()
        if i % (config['val_freq'] * (config['batch_size'] // config['dataloader_batch'])) == 0:
            writer.add_scalar('loss/train', total_loss / count, epoch)
            context_encoder.eval()
            query_encoder.eval()
            predictor.eval()
            with torch.no_grad():
                loss = 0
                target_to_pred = {}
                target_to_real = {}
                all_pred = []
                all_real = []
                for j, (context_x, context_y, query_x, query_y, targets) in enumerate(test_dataloader):
                    context_x, context_y, query_x, query_y = context_x.to(dtype=torch.float32, device='cuda'), context_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1), query_x.to(dtype=torch.float32, device='cuda'), query_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1)
                    context = torch.zeros((len(config['context_ranges']), len(context_x), config['d_model']), device='cuda')
                    for k in range(len(config['context_ranges'])):
                        context[k] = context_encoder(context_x[:, k, :], context_y[:, k, :])
                    context = context.mean(0)
                    query = query_encoder(query_x)
                    x = torch.concat((context, query), dim=1)
                    out = predictor(x)
                    loss += torch.mean((out - query_y) ** 2).item()
                    pred = out.cpu().numpy().flatten()
                    real = query_y.cpu().numpy().flatten()
                    all_pred.extend(pred)
                    all_real.extend(real)
                    for k, target in enumerate(targets):
                        if target not in target_to_real:
                            target_to_pred[target] = []
                            target_to_real[target] = []
                        target_to_pred[target].append(pred[k])
                        target_to_real[target].append(real[k])
                writer.add_scalar('loss/test', loss / (j + 1), epoch)
                try:
                    writer.add_scalar('corr/raw', linregress(all_pred, all_real).rvalue, epoch)
                except:
                    writer.add_scalar('corr/raw', 0, epoch)
                corrs = []
                for target in target_to_real:
                    try:
                        corrs.append(linregress(target_to_pred[target], target_to_real[target]).rvalue)
                    except:
                        corrs.append(0)
                writer.add_scalar('corr/per_target', np.mean(corrs), epoch)

            context_encoder.train()
            query_encoder.train()
            predictor.train()
        if i % (config['batch_size'] // config['dataloader_batch']) == 0:
            total_loss = 0
            count = 0
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            epoch += 1
            if epoch == config['total_epochs']:
                torch.save((context_encoder.state_dict(), query_encoder.state_dict(), predictor.state_dict()), f'model.pt')
                exit()
