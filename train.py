import torch
from torch import optim
from scipy.stats import linregress
import numpy as np
from models import *
from data import *
from torch.utils.tensorboard import SummaryWriter
import argparse



def train_fscap(context_num, data_file, args):
    config = {
        'run_name': args.run_name,
        'context_ranges': [(-50, 50)] * context_num,
        'dataloader_batch': None,
        'val_freq': 4096,
        'checkpoint_every_n_vals': 4,
        'lr': 0.00005,
        'batch_size': 1024,
        'warmup_steps': 128,
        'encoding_dim': 512,
        'total_epochs': int(2 ** 17),
    }
    config['dataloader_batch'] = 1024 // len(config['context_ranges'])


    train_dataloader, test_dataloader = get_dataloaders(config['dataloader_batch'], config['context_ranges'], data_file)
    context_encoder = Encoder(2048, config).cuda()
    query_encoder = Encoder(2048, config).cuda()
    predictor = Predictor(config['encoding_dim'] * 2, config).cuda()

    optimizer = optim.Adam(list(context_encoder.parameters()) + list(query_encoder.parameters()) + list(predictor.parameters()), lr=config['lr'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, 0.0001, 1, total_iters=config['warmup_steps'])
    annealing_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['total_epochs'])
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, annealing_scheduler], milestones=[config['warmup_steps']])
    writer = SummaryWriter(f"./runs/{config['run_name']}")

    epoch = 0
    ckpt_counter = 0
    while True:
        total_loss = 0
        count = 0
        for i, (context_x, context_y, query_x, query_y, _) in enumerate(train_dataloader):
            context_x, context_y, query_x, query_y = context_x.to(dtype=torch.float32, device='cuda'), context_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1), query_x.to(dtype=torch.float32, device='cuda'), query_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1)
            context = torch.zeros((len(config['context_ranges']), len(context_x), config['encoding_dim']), device='cuda')
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
                        context = torch.zeros((len(config['context_ranges']), len(context_x), config['encoding_dim']), device='cuda')
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
                    corrs = []
                    for target in target_to_real:
                        corrs.append(linregress(target_to_pred[target], target_to_real[target]).rvalue)
                    writer.add_scalar('per_target/mean', np.mean(corrs), epoch)

                    ckpt_counter += 1
                    if ckpt_counter == config['checkpoint_every_n_vals']:
                        torch.save((context_encoder.state_dict(), query_encoder.state_dict(), predictor.state_dict()), args.checkpoint_file)
                        ckpt_counter = 0

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
                    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_context_compounds', type=int, default=8)
    parser.add_argument('--dataset_name', type=str, default='bindingdb')
    parser.add_argument('--run_name', type=str, default='fscap_run')
    parser.add_argument('--checkpoint_file', default='model.pt')
    args = parser.parse_args()
    train_fscap(args.num_context_compounds, args.dataset_name, args)