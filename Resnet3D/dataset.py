import os
import glob
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import v2 as Tv2
    
class Video_dataset(Dataset):
    def __init__(self, opt, root):
            super().__init__()

            self.samples = []
            if opt.split == 'train':
                self.augment = True
                self.transform = Tv2.Compose(
                    [
                        Tv2.ToDtype(torch.float32, scale=True),
                        Tv2.RandomApply([Tv2.RandomResizedCrop(300, scale=(0.9,1.1), ratio=(3/4,4/3))], 0.2),
                        Tv2.RandomVerticalFlip(p=0.5),
                        Tv2.RandomHorizontalFlip(p=0.5),
                        Tv2.RandomCrop((256,256)),
                        Tv2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
                    ]
                )
            else:
                self.augment = False
                self.transform = Tv2.Compose(
                    [
                        Tv2.ToDtype(torch.float32, scale=True),
                        Tv2.CenterCrop((256,256)),
                        Tv2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
                    ]
                )

            self.dilation = opt.dilation
            self.inverse = opt.invert_labels
            self.n_frames = opt.n_frames
            self.num_batches = opt.num_batches

            if opt.split is not None:
                with open(f'media/{opt.split}.txt') as f:
                    videos = [line.replace('\n', '') for line in f.readlines()]
            else:
                videos = []

            self.videos = []
            
            for subdir in os.listdir(root):
                if opt.split is not None and subdir not in videos:
                    continue
                
                limit = 105 if 'clips_original' in root else 1000
                self.samples.append([sorted(glob.glob(f'{root}/{subdir}/*.png'))[:limit], (1.0 - self.inverse) if ('clips_original' in root or 'original_size' in root) else (1.0 * self.inverse)])
                self.videos.append(f'{root}/{subdir}')

                if opt.split == 'train' and 'clips_original' in root:
                    self.samples.append([sorted(glob.glob(f'{root}/{subdir}/*.png'))[:limit], (1.0 - self.inverse) if ('clips_original' in root or 'original_size' in root) else (1.0 * self.inverse)])
                    self.videos.append(f'{root}/{subdir}')


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        paths, target = self.samples[index]
        video = self.videos[index]

        if not paths:
            raise ValueError("No image files found in the folder.")

        in_tens_arr_batch = []

        for i in range(self.num_batches):
            if self.augment:
                start_frame = random.randint(0, len(paths) - (self.n_frames - 1) * self.dilation - 1)
            else:
                # uniform sampling from 0 to len(paths) - (self.n_frames - 1) * self.dilation - 1) based on self.num_batches, if self.num_batches < len(paths) - (self.n_frames - 1) * self.dilation - 1 restart from 0
                viable_start_frames = list(range(0, len(paths) - (self.n_frames - 1) * self.dilation - 1))
                offset = len(viable_start_frames) // self.num_batches
                start_frame = viable_start_frames[(i * offset) % len(viable_start_frames)]
        
            frame_indices = [start_frame + i * self.dilation for i in range(self.n_frames) if start_frame + i * self.dilation < len(paths)]

            in_tens_arr = []
            for idx in frame_indices:
                img = Image.open(paths[idx]).convert("RGB")
                img = Tv2.ToImage()(img)
                in_tens_arr.append(img.unsqueeze(0)) 
            
            in_tens = torch.cat(in_tens_arr, 0)

            if self.augment and random.random() < 0.3:
                if random.random() < 0.5:
                    sigma = random.uniform(0,3)
                    in_tens = Tv2.functional.gaussian_blur(in_tens, kernel_size=15, sigma=sigma)
                    
                else:
                    quality = random.randint(65,95)
                    in_tens = Tv2.functional.jpeg(in_tens, quality)

            if self.augment and random.random() < 0.2:
                in_tens = Tv2.functional.gaussian_blur(in_tens.permute(1,2,3,0), kernel_size=(1,3)).permute(3,0,1,2)

            x = self.transform(in_tens)

            if self.num_batches == 1:
                return x.permute(1,0,2,3), target, video
            else:
                in_tens_arr_batch.append((x.permute(1,0,2,3), target, video))

        return in_tens_arr_batch

def create_dataloader(opt, subdir='.', is_train=True):
    if subdir == "train":
        techniques = ["DynamiCrafter", "TokenFlow"]
        datasets = ["640x360_gen"]
        opt.split = 'train'
        opt.batch_size = max(1, opt.batch_size // opt.num_batches)

    elif subdir == "val":
        techniques = ["DynamiCrafter", "TokenFlow"]
        datasets = ["640x360_gen"]
        opt.split = 'val'
        opt.batch_size = max(1, opt.batch_size // opt.num_batches)

    elif subdir == "test":
        techniques = ["DynamiCrafter", "TokenFlow", "RAVE", "SEINE", "Text2Video-Zero"]
        datasets = ["640x360_gen", "640x360_gen_23", "640x360_gen_30", "640x360_gen_50"]
        opt.split = 'test'
        opt.batch_size = max(1, opt.batch_size // opt.num_batches)

    dset_lst = []
    for technique in techniques:
        print(technique)
        for dataset in datasets:
            root = os.path.join(opt.data_root, technique, dataset)
            dset = Video_dataset(opt, root)
            dset_lst.append(dset)

    print('clip_original')
    for dataset in datasets:
        root = os.path.join(opt.data_root, 'clips_original', dataset.replace("_gen", ""))
        dset = Video_dataset(opt, root)
        dset_lst.append(dset)

    # commercial dataset
    if subdir == "test":
        for technique in ["Cogvideo", "SORA", "LUMA_AI", "Hunyuan", "RunawayML", "original_size"]:
            print(technique)
            opt.split = None
            root = os.path.join(opt.data_root_commercial, technique, 'frames')
            dset = Video_dataset(opt, root)
            dset_lst.append(dset)

    dataset = torch.utils.data.ConcatDataset(dset_lst)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=int(opt.num_threads),
        shuffle = True if is_train else False,
        collate_fn=(lambda x: default_collate([p for v in x for p in v])) if opt.num_batches > 1 else None,
    )
    return data_loader
