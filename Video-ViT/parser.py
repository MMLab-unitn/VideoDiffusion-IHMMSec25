import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test", help="run name")

    parser.add_argument("--device", type=str, default="cuda:0", help="cuda device to use")
    parser.add_argument("--model", type=str, default="vit_head", help="architecture name")
    parser.add_argument("--freeze", action='store_true', help="Freeze all layers except the last one")
    parser.add_argument("--prototype", action='store_true', help="Use prototype layer")
    parser.add_argument("--invert_labels", action='store_true', help="Inverted labels for prototype")

    parser.add_argument("--num_epochs", type=int, default=200, help="# of epoches at starting learning rate")

    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--lr_decay_epochs",type=int, default=5, help="Number of epochs without loss reduction before lowering the learning rate by 10x")
    parser.add_argument("--lr_min",type=float, default=1e-7, help="minimum learning rate")

    parser.add_argument("--split_path", type=str, help="Path to split files")
    parser.add_argument("--data_root", type=str, default="/media/mmlab/Datasets_4TB/videodiffusion", help="Path to dataset")
    parser.add_argument("--data_root_commercial", type=str, default="/media/mmlab/Datasets_4TB/videodiffusion_extra", help="Path to dataset for commercial tools")

    parser.add_argument("--batch_size", type=int, default=32, help='Dataloader batch size')
    parser.add_argument("--num_threads", type=int, default=24, help='# threads for loading data')

    parser.add_argument("--n_frames", type=int, default=8, help='Number of frames for the ViT')
    parser.add_argument("--dilation", type=int, default=1, help='Dilation of frames sampling')
    parser.add_argument("--num_batches", type=int, default=8, help='Number of batches of frames for each video, BEWARE, this options effectively multiplies the number of samples in the dataset')

    return parser