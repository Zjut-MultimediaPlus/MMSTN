import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
from matplotlib import animation

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default=r'G:\software\code\sgan-master-u-withME\scripts\tymodel\visual', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

xdata, ydata = [], []
x1=[]
y1=[]
aa, bb = [], []
fig, ax = plt.subplots()
ln, = ax.plot([], [], 'ro')

def gen_dot():
    for i in range(0,len(x1)):
        newdot = [x1[i], y1[i]]
        yield newdot

def update_dot(newd):
    xdata.append(newd[0])
    ydata.append(newd[1])
    ln.set_data(xdata, ydata)
    return ln,

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate(args, loader, generator, num_samples,modelPath):
    with torch.no_grad():
        plotCount = 0
        for batch in loader:
            tyID = batch[-1]
            batch = [tensor.cuda() for tensor in batch[:-1]]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end,obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me) = batch
            gt = pred_traj_gt[:, :, :].data
            input_a = obs_traj[:, :, :].data
            aa = np.concatenate((input_a.cuda().data.cpu().numpy(), gt.cuda().data.cpu().numpy()), axis=0)
            pred_list = []
            obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
            # pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
            obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
            for _ in range(3):#num_samples
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                pred_traj_fake = pred_traj_fake[:, :, :2]
                pred_list.append(pred_traj_fake)
            for i in range(aa.shape[1]):
                plotCount += 1
                for pred_traj_fakex in pred_list:
                    out_a=pred_traj_fakex[:,i,:].data
                    bb=np.concatenate((input_a[:, i, :].cuda().data.cpu().numpy(),out_a.cuda().data.cpu().numpy()),axis=0)
                    # global x1,y1
                    plt.xlim(-10, 10)
                    plt.ylim(-6, 6)
                    x1=bb[:,0]
                    y1=bb[:,1]
                    l2 = plt.plot(x1, y1, '.')
                # ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot, interval = 5)
                l = plt.plot(aa[:,i, 0], aa[:,i, 1], '.', color='red')
                _,filename = os.path.split(modelPath)
                fileN,_ = os.path.split(filename)
                savePath = os.path.join(r'G:\software\code\sgan-master-u-withME\scripts\tymodel\plot',fileN)
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                name = tyID[i][0][0]+str(tyID[i][0][1])
                plt.savefig(os.path.join(savePath,name+'.png'))
                # plt.show()
                plt.close()
def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        modelPath = path
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        evaluate(_args, loader, generator, args.num_samples,modelPath)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
