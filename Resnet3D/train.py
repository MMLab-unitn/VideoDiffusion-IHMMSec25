# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import glob
import torch
import shutil
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from networks import get_network
from parser import get_parser
from dataset import create_dataloader

# ----------------------------------------------------------------------------
# DA INVERTIRE LABEL PER PROTOTYPES
# ----------------------------------------------------------------------------

def check_accuracy(loader, model, settings):
    model.eval()

    label_array = torch.empty(0, dtype=torch.int64, device=device)
    pred_array = torch.empty(0, dtype=torch.int64, device=device)
    
    with torch.no_grad():
        with tqdm(loader, unit='batch', mininterval=0.5) as tbatch:
            tbatch.set_description(f'Validation')
            for (data, label, _) in tbatch:
                data = data.to(device)
                label = label.to(device)
                
                scores = model(data).squeeze(1)

                pred = torch.round(torch.sigmoid(scores)).int()

                label_array = torch.cat((label_array, label))
                pred_array = torch.cat((pred_array, pred))

                zerosamples = torch.count_nonzero(label_array==0)*2
                onesamples = torch.count_nonzero(label_array==1)
                totalsamples = zerosamples + onesamples

                zerocorrect = torch.count_nonzero(pred_array[label_array==0]==0)*2
                onecorrect = torch.count_nonzero(pred_array[label_array==1]==1)
                totalcorrect = zerocorrect + onecorrect

                zeroaccuracy = float(zerocorrect/zerosamples)
                oneaccuracy = float(onecorrect/onesamples)
                totalaccuracy = float(totalcorrect/totalsamples)
                
                tbatch.set_postfix(acc_tot=totalaccuracy*100, acc_fake=zeroaccuracy*100, acc_real=oneaccuracy*100)

        print(f'Got accuracy {float(totalcorrect)/float(totalsamples)*100:.2f} \n')
    return totalaccuracy


def train(loader, val_dataloader, model, settings):
    best_accuracy = 0
    lr_decay_counter = 0
    for epoch in range(0, settings.num_epochs):
        model.train()
        with tqdm(loader, unit='batch', mininterval=0.5) as tepoch:
            tepoch.set_description(f'Epoch {epoch}', refresh=False)
            for batch_idx, (data, label, _) in enumerate(tepoch):
                data = data.to(device)
                label = label.to(device)

                scores = model(data).squeeze(1)

                loss = criterion(scores, label).mean()
 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item())

        accuracy = check_accuracy(val_dataloader, model, settings)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'./train/{settings.name}/models/best.pt')

            print(f'New best model saved with accuracy {best_accuracy:.2f} \n')
            lr_decay_counter = 0

        elif settings.lr_decay_epochs > 0:
            lr_decay_counter += 1
            if lr_decay_counter == settings.lr_decay_epochs:
                if optimizer.param_groups[0]['lr'] > settings.lr_min:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1
                    print('Learning rate decayed \n')
                    lr_decay_counter = 0
                else:
                    print('Learning rate already at minimum \n')
                    break
        

if __name__ == "__main__":
    parser = get_parser()
    settings = parser.parse_args()
    print(settings)

    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

    os.makedirs(f'./train/{settings.name}/models', exist_ok=True)
    for file in glob.glob(f'*.py'):
        shutil.copy(file, f'./train/{settings.name}/')
    
    with open(f'./train/{settings.name}/settings.txt', 'w') as f:
        f.write(str(settings))

    train_dataloader = create_dataloader(settings, subdir='train', is_train=True)
    val_dataloader = create_dataloader(settings, subdir='val', is_train=False)


    model = get_network(settings)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=settings.lr)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    train(train_dataloader, val_dataloader, model, settings)