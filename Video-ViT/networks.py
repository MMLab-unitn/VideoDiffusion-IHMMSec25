import sys
sys.path.append('ViT_panda')
from ViT_panda.get_features import get_extractor as get_extractor_panda

import torch.nn as nn
import torch

class ScoresLayer(nn.Module):
    def __init__(self, input_dim, num_centers):
        super().__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.zeros(num_centers, input_dim), requires_grad=True)
        self.logsigmas = nn.Parameter(torch.zeros(num_centers), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(batch_size, self.input_dim, 1, 1) # [batch, C, 1, 1]

        centers = self.centers[None, :, :, None, None]  # [1, K, C, 1, 1]
        diff = out.unsqueeze(1) - centers  # [batch, K, C, 1, 1]

        sum_diff = torch.sum(diff, dim=2)  # [batch, K, 1, 1]
        sign = torch.sign(sum_diff)

        squared_diff = torch.sum(diff ** 2, dim=2)  # [batch, K, 1, 1]

        logsigmas = nn.functional.relu(self.logsigmas)
        denominator = 2 * torch.exp(2 * logsigmas)
        part1 = (sign * squared_diff) / denominator.view(1, -1, 1, 1)

        part2 = self.input_dim * logsigmas
        part2 = part2.view(1, -1, 1, 1)

        scores = part1 + part2
        output = scores.sum(dim=(1, 2, 3)).view(-1, 1)  # [batch, 1]

        return output
    
def get_network(settings):
    name = settings.model

    if name == 'vit_panda':
        class VideoClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 8 ) # 32 x 4096 > 8 x 4096
                self.fc2 = nn.Linear(4096, 512) # 8 x 4096 > 8 x 512
                self.fc3 = nn.Linear(4096, 1) # 8 x 512 > 1
                if settings.prototype:
                    self.proto = ScoresLayer(input_dim=self.fc3.out_features, num_centers=1)

            def forward(self, x):
                x = self.fc1(x.permute(0,2,1)).permute(0,2,1)
                x = self.fc2(x).flatten(1)
                x = self.fc3(x)
                if settings.prototype:
                    x = self.proto(x)
                return x

        model = VideoClassifier()

        for param in model.parameters():
            param.requires_grad = True

        extractor = get_extractor_panda(settings)
    
    else:
        raise NotImplementedError('model not recognized')

    return model, extractor


