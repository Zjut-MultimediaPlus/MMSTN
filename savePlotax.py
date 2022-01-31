import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
import cv2
from matplotlib import animation
import matplotlib.image as img
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default=r'G:\software\code\TC-Prediction\sgan-master-u-withME - 13\scripts\tymodel\visual', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test2019', type=str)

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

def toNE(pred_traj,pred_Me):
    # 0  经度  1纬度
    pred_traj[:, :,0] = (pred_traj[:, :,0] / 10 * 500 + 1300)/10
    pred_traj[:,:,1] = (pred_traj[:,:,1] / 6 * 300 + 300)/10
    # 0 气压 1 风速
    pred_Me[:, :, 0] = pred_Me[:, :, 0]*50+960
    pred_Me[:, :, 1] = pred_Me[:, :, 1] * 25 + 40
    return pred_traj,pred_Me

def getMaxMinXY(pred_last):
    x = pred_last[:,:,0]
    y = pred_last[:,:,1]
    xmaxindex = np.argmax(x,axis=1)
    xminindex = np.argmin(x,axis=1)
    xmax = np.max(x, axis=1)
    xmin = np.min(x, axis=1)
    xindex = np.stack((xminindex,xmaxindex)).transpose(1,0)
    xminmax = np.stack((xmin,xmax)).transpose(1,0)
    return xindex,xminmax

def getPossibelarea(xmin,xmax,startx,ymin,ymax,yend):
    medx = startx[0]
    yendindex = np.argsort(yend[:,0])
    yendsord = yend[yendindex]
    yminaddstart = np.vstack([startx,ymin])
    ymaxaddstart = np.vstack([startx, ymax])
    funcmin = interp1d(yminaddstart[:, 0], yminaddstart[:, 1], kind='slinear')
    funcmax = interp1d(ymaxaddstart[:, 0], ymaxaddstart[:, 1], kind='slinear')
    funcend = interp1d([yendsord[0, 0],yendsord[-1, 0]], [yendsord[0, 1],yendsord[-1, 1]], kind='slinear')


    if medx > xmax:
        x1 = np.linspace(xmin, xmax, 300)
        x2 = np.linspace(xmax, medx, 300)
        # y1bottom = make_interp_spline(yminaddstart[:, 0], yminaddstart[:, 1])(x1)
        # y1up = make_interp_spline(yendsord[:, 0], yendsord[:, 1])(x1)
        # y2bottom = make_interp_spline(yminaddstart[:, 0], yminaddstart[:, 1])(x2)
        # y2up = make_interp_spline(ymaxaddstart[:, 0], ymaxaddstart[:, 1])(x2)
        y1bottom = funcmin(x1)
        y2bottom = funcmin(x2)
        y1up = funcend(x1)
        y2up = funcmax(x2)
    elif medx < xmin:
        x1 = np.linspace(medx, xmin, 300)
        x2 = np.linspace(xmin, xmax, 300)
        # y1bottom = make_interp_spline(ymaxaddstart[:, 0], ymaxaddstart[:, 1])(x1)
        # y1up = make_interp_spline(yminaddstart[:, 0], yminaddstart[:, 1])(x1)
        # y2bottom = make_interp_spline(ymaxaddstart[:, 0], ymaxaddstart[:, 1])(x2)
        # y2up = make_interp_spline(yendsord[:, 0], yendsord[:, 1])(x2)
        y1bottom = funcmax(x1)
        y2bottom = funcmax(x2)
        y1up = funcmin(x1)
        y2up = funcend(x2)
    else:
        x1 = np.linspace(xmin, medx, 300)
        x2 = np.linspace(medx, xmax, 300)
        # y1bottom = make_interp_spline(yminaddstart[:, 0], yminaddstart[:, 1])(x1)
        # y1up = make_interp_spline(yendsord[:, 0], yendsord[:, 1])(x1)
        # y2bottom = make_interp_spline(ymaxaddstart[:, 0], ymaxaddstart[:, 1])(x2)
        # y2up = make_interp_spline(yendsord[:, 0], yendsord[:, 1])(x2)
        y1bottom = funcmin(x1)
        y2bottom = funcmax(x2)
        y1up = funcend(x1)
        y2up = funcend(x2)
    # plt.plot(x1,y1bottom,color='r')
    # plt.plot(x1, y1up, color='r')
    # plt.plot(x2, y2bottom, color='g')
    # plt.plot(x2, y2up, color='g')
    # plt.show()
    return {'x1':x1,'x2':x2,'y1up':y1up,'y1bottom':y1bottom,'y2up':y2up,'y2bottom':y2bottom}

def getPicName(tyid):
    tyname = tyid[0]['new'][1]
    date = tyid[0]['new'][0]
    root = r'G:\data\Typhoon\TY2019_img'
    datef = date[:-2]+'_'+date[-2:]
    filelist = os.listdir(os.path.join(root,tyname))
    for filename in filelist:
        if datef in filename:
            return os.path.join(root,tyname,filename)
    return 0

def getclosetra(i,gt,pre):
    pres = torch.stack(pre).data.cpu().numpy()
    pres = pres[:,:,i,:]
    x = pres[:,:,0]-gt[:,0]
    y = pres[:,:,1]-gt[:,1]
    dist = x**2+y**2
    sumdist = np.sum(dist,axis=1)
    mindist = np.min(sumdist)
    mask = (sumdist==mindist)*1
    return mask



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

            pred_list,pred_list_Me = [],[]
            obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
            pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
            obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
            obs_traj_real, obs_traj_Me_real = toNE(obs_traj.cuda().data.cpu().numpy()[:, :, :2],obs_traj.cuda().data.cpu().numpy()[:, :, 2:])
            pred_traj_gt_real,pred_traj_gt_real_Me = toNE(pred_traj_gt.cuda().data.cpu().numpy()[:, :, :2],pred_traj_gt.cuda().data.cpu().numpy()[:, :, 2:])
            aa = np.concatenate((obs_traj_real, pred_traj_gt_real), axis=0)
            for _ in range(6):#num_samples
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake_all = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                pred_traj_fake = pred_traj_fake_all[:4, :, :2]
                pred_traj_fake_Me = pred_traj_fake_all[:4, :, 2:]
                pred_traj_fake,pred_traj_fake_Me = toNE(pred_traj_fake,pred_traj_fake_Me)
                pred_list.append(pred_traj_fake)
                pred_list_Me.append(pred_traj_fake_Me)

            pred_last = torch.stack(pred_list).data.cpu().numpy()[:,-1,:,:].transpose(1,0,2)
            xindex,xminmax = getMaxMinXY(pred_last)
            color = ['#47F8FC','#8352FF','#38FF39','#FF741F','#FFEF2B','steelblue']
            # bgimg = img.imread(r'G:\CreateDateset\airmass2.png')
            # bgimg = img.imread('tm.png')
            # bgimg = cv2.resize(bgimg,(1000,1000))
            for i in range(aa.shape[1]):
                name = tyID[i][0]['new'][1] + str(tyID[i][0]['new'][0])
                if name != 'HALONG2019110800':
                    continue
                if getPicName(tyID[i])==0:
                    print(i)
                    continue
                bgimg = img.imread(getPicName(tyID[i]))
                plotCount += 1
                fig = plt.figure(figsize=(10, 10))
                fig.figimage(bgimg)
                ax = fig.add_axes([0, 0, 1, 1])
                # ax.axesPatch.set_alpha(0.05)
                # ax.set_axisbelow(True)

                ax.set_xlim(80, 200)
                ax.set_ylim(-60, 60)
                traplot = [1,1,1,1,1,1]

                traplot = getclosetra(i,aa[12:16,i, :],pred_list)
                color = ['#38FF39', '#38FF39', '#38FF39', '#38FF39', '#38FF39', '#38FF39']

                for j,pred_traj_fakex in enumerate(pred_list):
                    if traplot[j] == 0:
                        continue
                    out_a=pred_traj_fakex[:,i,:].data
                    bb=np.concatenate((input_a[:, i, :].cuda().data.cpu().numpy(),out_a.cuda().data.cpu().numpy()),axis=0)
                    # global x1,y1
                    x1=bb[:,0]
                    y1=bb[:,1]
                    ax.plot(x1, y1, '*',markersize=5,color=color[j])
                # plt.show()
                # ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot, interval = 5)
                ax.plot(aa[:16,i, 0], aa[:16,i, 1], '.', color='red',markersize=5)
                # 覆盖区域===============
                xmin = xminmax[i,0]
                xmax = xminmax[i,1]
                startx = obs_traj_real[-1, i, :]
                ymin = pred_list[xindex[i,0]][:,i,:].data.cpu().numpy()
                ymax = pred_list[xindex[i, 1]][:, i, :].data.cpu().numpy()
                yend = pred_last[i]
                areadict = getPossibelarea(xmin,xmax,startx,ymin,ymax,yend)
                x = np.append(areadict['x1'],areadict['x2'])
                y1 = np.append(areadict['y1up'],areadict['y2up'])
                y2 = np.append(areadict['y1bottom'],areadict['y2bottom'])
                ax.fill_between(x, y1, y2, color='#F52100', alpha=.5)  #color='#F52100'
                ax.set_zorder(100)
                ax.set_axis_off()
                # plt.figimage(bgimg)
                # plt.axis('off')
                # plt.fill_between(areadict['x1'],areadict['y1up'],areadict['y1bottom'],color='blue',alpha=.25)
                # plt.fill_between(areadict['x2'], areadict['y2up'], areadict['y2bottom'], color='blue', alpha=.25)



                _,filename = os.path.split(modelPath)
                fileN,_ = os.path.split(filename)
                savePath = os.path.join(r'G:\software\code\TC-Prediction\sgan-master-u-withME - 13\scripts\tymodel\plot',fileN)
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                name = tyID[i][0]['new'][1]+str(tyID[i][0]['new'][0])
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
