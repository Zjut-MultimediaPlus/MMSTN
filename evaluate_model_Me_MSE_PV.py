import argparse
import os
import torch
import copy

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error,toNE,trajectory_displacement_error,value_error,trajectory_diff,value_diff
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default='models/decoder-轨迹没有更新', type=str)
parser.add_argument('--num_samples', default=6, type=int)
parser.add_argument('--dset_type', default='test', type=str)


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


def getmin_helper(error,an,timeanpv, seq_start_end,timeanNE,timeanabsPV):
    sum_ = 0
    error = torch.stack(error, dim=1)
    an = torch.stack(an, dim=1)
    timeanpv = torch.stack(timeanpv, dim=1)
    timeanNE = torch.stack(timeanNE, dim=1)
    timeanabsPV = torch.stack(timeanabsPV, dim=1)
    minpoint,minpoint_pv,minpoint_NE,minpoint_absPV = [],[],[],[]


    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.argmin(_error)
        # _error = torch.mean(_error)
        minpoint.append(an[start:end,_error.data.cpu()])
        minpoint_pv.append(timeanpv[start:end,_error.data.cpu()])
        minpoint_NE.append(timeanNE[start:end, _error.data.cpu()])
        minpoint_absPV.append(timeanabsPV[start:end, _error.data.cpu()])

    minpoint = torch.stack(minpoint,dim=1).squeeze()
    minpoint_pv = torch.stack(minpoint_pv,dim=1).squeeze()
    minpoint_NE = torch.stack(minpoint_NE, dim=1).squeeze()
    minpoint_absPV = torch.stack(minpoint_absPV, dim=1).squeeze()
    return {'tr':minpoint,'pv':minpoint_pv,'ne':minpoint_NE,'absPV':minpoint_absPV}

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        # _error = torch.mean(_error)

        sum_ += _error
    return sum_

def ve_evaluate_helper(error, seq_start_end):
    sum_p = 0
    sum_v = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error,dim=0)
        # _error = _error[0]
        # _error = torch.mean(_error,dim=0)
        sum_p += _error[0][0]
        sum_v += _error[0][1]
    return sum_p,sum_v

def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer,tde_outer,ve_outer,ana_outer,pv_outer,gt = [], [],[],[],[],[],[]
    neout = []
    abspvout = []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch[:-1]]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end,obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me) = batch

            ade, fde,tde,ve = [], [],[],[]
            analyse,analyse_pv,analyseNE,analyseabsPV = [],[],[],[]
            total_traj += pred_traj_gt.size(1)
            obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
            gt.append(torch.cat([pred_traj_gt.permute(1, 0, 2), pred_traj_gt_Me.permute(1, 0, 2)], dim=2))
            # pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
            obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
            pred_traj_gt_rel = torch.cat([pred_traj_gt_rel, pred_traj_gt_rel_Me], dim=2)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake_relt = pred_traj_fake_rel
                pred_traj_fake_rel = pred_traj_fake_relt[:,:,:2]
                pred_traj_fake_rel_Me = pred_traj_fake_relt[:,:,2:]

                # pred_traj_fake_rel 用来预测后12个点与第8点的偏差
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1,:,:2]
                )
                pred_traj_fake_rel_Me = relative_to_abs(
                    pred_traj_fake_rel_Me, obs_traj_Me[-1]
                )
                # 只看坐标的偏差
                real_pred_traj_fake,real_pred_traj_fake_Me = toNE(pred_traj_fake,pred_traj_fake_rel_Me)
                # 函数会改变参数变量
                real_pred_traj_gt,real_pred_traj_gt_Me = toNE(copy.deepcopy(pred_traj_gt),copy.deepcopy(pred_traj_gt_Me))
                real_pred_traj_fake = real_pred_traj_fake[:, :, :2]

                # ade.append(displacement_error(
                #     pred_traj_fake, pred_traj_gt, mode='raw'
                # ))
                analyseNE.append(real_pred_traj_fake.permute((1,0,2)))
                analyseabsPV.append(real_pred_traj_fake_Me.permute((1, 0, 2)))
                analyse.append(trajectory_diff(
                    real_pred_traj_fake, real_pred_traj_gt, mode='raw'
                ))
                analyse_pv.append(value_diff(
                    real_pred_traj_fake_Me, real_pred_traj_gt_Me, mode='raw'
                ))
                tde.append(trajectory_displacement_error(
                    real_pred_traj_fake, real_pred_traj_gt, mode='raw'
                ))
                ve.append(value_error(
                    real_pred_traj_fake_Me, real_pred_traj_gt_Me, mode='raw'
                ))

                # fde.append(final_displacement_error(
                #     pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                # ))
            time_tde_sum = []
            time_ve_sum = []
            time_an_sum = []
            time_anpv_sum = []
            time_anne_sum = []
            time_anabsPV_sum = []

            for i in range(args.pred_len):
                timeade = [x[:,i] for x in tde]
                timean = [x[:, i] for x in analyse]
                timeanpv = [x[:, i] for x in analyse_pv]
                timeanNE = [x[:, i] for x in analyseNE]
                timeanabsPV = [x[:, i] for x in analyseabsPV]
                out = getmin_helper(timeade,timean,timeanpv, seq_start_end,timeanNE,timeanabsPV)
                time_an_sum.append(out['tr'])
                time_anpv_sum.append(out['pv'])
                time_anne_sum.append(out['ne'])
                time_anabsPV_sum.append(out['absPV'])

            for i in range(args.pred_len):
                timeade = [x[:,i] for x in tde]
                time_tde_sum.append(evaluate_helper(timeade, seq_start_end))

            for i in range(args.pred_len):
                timeade = [x[:,i] for x in ve]
                time_ve_sum.append(ve_evaluate_helper(timeade, seq_start_end))
            # ade_sum = evaluate_helper(ade, seq_start_end)
            # fde_sum = evaluate_helper(fde, seq_start_end)

            tde_outer.append(time_tde_sum)
            ve_outer.append(time_ve_sum)
            time_an_sum = torch.stack(time_an_sum,dim=1)
            ana_outer.append(time_an_sum)
            time_anpv_sum = torch.stack(time_anpv_sum, dim=1)
            pv_outer.append(time_anpv_sum)
            time_anne_sum = torch.stack(time_anne_sum, dim=1)
            neout.append(time_anne_sum)
            time_anabsPV_sum = torch.stack(time_anabsPV_sum, dim=1)
            abspvout.append(time_anabsPV_sum)
            # fde_outer.append(fde_sum)
        # ade_outer = torch.stack(ade_outer, dim=1)cvpr
        saveana(ana_outer,pv_outer,gt,neout,abspvout)
        tde_outer = torch.tensor(tde_outer)
        ve_outer = torch.tensor(ve_outer)
        ade = torch.sum(tde_outer,dim=0) / (total_traj)
        ve = torch.sum(ve_outer,dim=0) / (total_traj)
        fde = 0
        return ade, ve

import numpy as np
def saveana(ana_outer,ve_outer,gt,neouter,abspvout):
    ana_outer = torch.cat(ana_outer, dim=0)
    ana_outer_np = ana_outer.data.cpu().numpy()
    ve_outer = torch.cat(ve_outer, dim=0)
    ve_outer_np = ve_outer.data.cpu().numpy()
    neouter = torch.cat(neouter, dim=0)
    neouter_np = neouter.data.cpu().numpy()
    abspvout = torch.cat(abspvout, dim=0)
    abspvout_np = abspvout.data.cpu().numpy()
    gt = torch.cat(gt, dim=0)
    gt_np = gt.data.cpu().numpy()
    if os.path.exists('trajectory.npy'):
        tra = np.load('trajectory.npy')
        np.save('trajectory.npy',(ana_outer_np+tra)/2)
    else:
        np.save('trajectory.npy', ana_outer_np)

    if os.path.exists('pvdif.npy'):
        pv = np.load('pvdif.npy')
        np.save('pvdif.npy', (ve_outer_np+pv)/2)
    else:
        np.save('pvdif.npy', ve_outer_np)

    if os.path.exists('ne.npy'):
        ne = np.load('ne.npy')
        np.save('ne.npy', (neouter_np+ne)/2)
    else:
        np.save('ne.npy', neouter_np)

    if os.path.exists('absPV.npy'):
        abspv = np.load('absPV.npy')
        np.save('absPV.npy', (abspvout_np+abspv)/2)
    else:
        np.save('absPV.npy', abspvout_np)

    np.save('gt.npy', gt_np)

    pass

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
        modelpath = path
        checkpoint = torch.load(path)
        # print(checkpoint['args'])
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path,test=True)
        tde, ve = evaluate(_args, loader, generator, args.num_samples)
        print(os.path.split(modelpath)[1])
        print('Dataset: {}, Pred Len: {}'.format(
            _args.dataset_name, _args.pred_len))
        print('TDR:',tde)
        print('TDR:', ve)
        return tde,ve



if __name__ == '__main__':
    args = parser.parse_args()
    num = 100
    for i in range(num):
        ted,ve = main(args)
        if i == 0:
            ated, ave = ted,ve
        else:
            ated += ted
            ave+=ve
    print(ated/num,ave/num)

# tensor([ 26.5824,  55.9043,  90.2042, 131.4698, 176.3860, 223.9666, 275.9354,
#         330.8814]) tensor([[2.0437, 1.0676],
#         [3.4686, 1.7413],
#         [4.6360, 2.3311],
#         [5.6804, 2.9030],
#         [6.5302, 3.3760],
#         [7.2055, 3.7640],
#         [7.6929, 4.0698],
#         [8.0324, 4.2986]])