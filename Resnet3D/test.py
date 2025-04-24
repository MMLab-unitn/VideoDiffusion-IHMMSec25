# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import torch
import pandas as pd
from tqdm import tqdm

from networks import get_network
from parser import get_parser
from dataset import create_dataloader

def test(loader, model, settings):
    model.eval()

    csv_filename = f'./train/{settings.name}/data/results.csv'
    df = pd.DataFrame(columns=['name', 'pro','flag'])
    
    with torch.no_grad():
        with tqdm(loader, unit='batch', mininterval=0.5) as tbatch:
            tbatch.set_description(f'Validation')
            for (data, labels, videos) in tbatch:
                data = data.to(device)
                labels = labels.to(device)

                scores = model(data).squeeze(1)

                for score, label, video in zip(scores, labels, videos):
                    df = df._append({'name': '/'.join(video.split('/')[-3:]),'pro': score.item(),'flag':label.item()}, ignore_index=True)

    df.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    parser = get_parser()
    settings = parser.parse_args()
    
    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

    os.makedirs(f'./train/{settings.name}/data', exist_ok=True)
    test_dataloader = create_dataloader(settings, subdir='test', is_train=False)

    model = get_network(settings)
    model.to(device)

    state_dict = torch.load(f'./train/{settings.name}/models/best.pt')
    model.load_state_dict(state_dict)

    test(test_dataloader, model, settings)