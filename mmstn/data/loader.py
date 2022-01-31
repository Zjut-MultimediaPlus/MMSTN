from torch.utils.data import DataLoader

# from sgan.data.trajectories import TrajectoryDataset, seq_collate
from mmstn.data.trajectoriesWithMe import TrajectoryDataset, seq_collate

def data_loader(args, path,test=None):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    if test is None:
        shuffle = True
    else:
        shuffle = False
        
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
