import torch
import torch.nn as nn

from pytorchvideo.models.hub import slow_r50

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

    if name == 'slow_3d':
        class VideoClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
                self.model.blocks[5].pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.model.blocks[5].proj = nn.Linear(self.model.blocks[5].proj.in_features, 1)
                self.model.dropout = None
                self.model.blocks[5].output_pool = None
                if settings.prototype:
                    self.proto = ScoresLayer(input_dim=self.model.blocks[5].proj.out_features, num_centers=1)

            def forward(self, x):
                x = self.model(x).squeeze().unsqueeze(1)
                if settings.prototype:
                    x = self.proto(x)
                return x
        
        model = VideoClassifier()

        if settings.freeze:
            for param in model.model.parameters():
                param.requires_grad = False
            for param in model.model.blocks[5].parameters():
                param.requires_grad = True

        else:
            for param in model.model.parameters():
                param.requires_grad = True

        for param in model.proto.parameters():
            param.requires_grad = True

    else:
        raise NotImplementedError('model not recognized')

    return model


