
####    The following is a piece of work that I have done for my Machine Visual Perception group project, 
####        where I load a dataset and prepare the formatting so that it can be easily retrieved from the DataLoader.

####    I appreciate this is probably unsuitable for demonstration, 
####        both in terms of 'it being unrelated' and 'it already being a separate piece of work'.


import os
import numpy as np
import imageio.v2 as imageio
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader


class DNerfDataset(Dataset):

    rays_rgb = []
    
    def __init__(self, datadir):


        splits = ['train', 'val', 'test']
        metas = {}
        all_imgs = []
        all_poses = []
        all_times = []
        counts = [0]

        for s in splits:
            with open(os.path.join(datadir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)
        
        for s in splits:
            meta = metas[s]

            imgs = []
            poses = []
            times = []
            # if s=='train' or testskip==0:
            #     skip = 2  # if you remove/change this 2, also change the /2 in the times vector
            # else:
            skip = 1
            
            for t, frame in enumerate(meta['frames'][::skip]):
                fname = os.path.join(datadir, frame['file_path'][2:] + '.png')
                #print(datadir)
                #print(fname)
                img = imageio.imread(fname)
                #print(img)
                imgs.append(img)
                poses.append(np.array(frame['transform_matrix']))
                cur_time = frame['time'] if 'time' in frame else float(t) / (len(meta['frames'][::skip])-1)
                times.append(cur_time)

            assert times[0] == 0, "Time must start at 0"

            imgs = (np.array(imgs)[:,:,:,:-1] / 255.).astype(np.float32)  # keep all 4 channels (RGBA), nope, now RGB
            poses = np.array(poses).astype(np.float32)
            times = np.array(times).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
            all_times.append(times)
    
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)] # isplit is a 3 element array, with each element being a list of indices for each split
        i_train, i_val, i_test = i_split
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        times = np.concatenate(all_times, 0)

        Hp, Wp = imgs[0].shape[:2] # Width in pixels # [:2]?
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * Wp / np.tan(.5 * camera_angle_x)
        
        

        #print(poses.shape)
        rays = np.stack([get_rays_np(Hp, Wp, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        #print(rays.shape)
        print('done, concats')
        rays_rgb = np.concatenate([rays, imgs[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        #print(rays_rgb.shape)
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
 

        # Get angles from direction
        directions = [item[1] for item in rays_rgb].astype(np.float32)
        angles = [(np.arccos(a/math.sqrt(a*a+b*b)), np.arccos(math.sqrt((a*a+b*b)/(a*a+b*b+c*c)))) for (a,b,c) in directions]
        
        assert len(angles) == len(rays_rgb)

        print('Example angles: ')
        print(angles[0]) # this is a tuple with (theta, phi)

        # the following does not work without NVidia GPU
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # rays_rgb = torch.Tensor(rays_rgb).to(device)
        # angles = torch.Tensor(rays_rgb).to(device)


    def __getitem__(self, index):
        ray = self.rays_rbg[index]
        angles = self.angles[index]
        return angles, ray

    def __len__(self):
        return len(self.rays_rgb)


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
 #   parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/hook/', help='input data directory')
    return parser



def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    return rays_o, rays_d





if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    datadir = args.datadir
    print(datadir)

    #train()
    traindataset = DNerfDataset(datadir)
    print(traindataset.__len__())
    #traindataset.__load__(datadir)
    train_loader = DataLoader(dataset=traindataset, batch_size=32, shuffle=False, num_workers=2)

    sampleangles, samplerays = next(iter(train_loader))
    print(f"Angles batch shape: {sampleangles.size()}")
    print(f"Rays batch shape: {samplerays.size()}")
    print('Sample angles: ')
    print(sampleangles)
    print('Sample rays: ')
    print(samplerays)

    print('DataLoader successfully loaded with rays')






# run the program, on some data



