import torch
import torch.nn as nn

class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()
        self.backbone = base_encoder                     # resnet, return 512
        prev_dim = self.backbone.fc.weight.shape[1]      # 512

        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                       nn.BatchNorm1d(prev_dim),
                                       nn.ReLU(inplace=True), # first layer
                                       nn.Linear(prev_dim, prev_dim, bias=False),
                                       nn.BatchNorm1d(prev_dim),
                                       nn.ReLU(inplace=True), # second layer
                                       self.backbone.fc,
                                       nn.BatchNorm1d(dim, affine=False)) # output layer    # 512
        self.projector[6].bias.requires_grad = False
        
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer


    def forward(self, x1, x2, x3, x4, y1, y2, y3, y4):

        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x3 = self.backbone(x3)
        x4 = self.backbone(x4)
        x = (x1+x2+x3+x4) / 4

        y1 = self.backbone(y1)
        y2 = self.backbone(y2)
        y3 = self.backbone(y3)
        y4 = self.backbone(y4)
        y = (y1+y2+y3+y4) / 4

        z1 = self.projector(x)
        z2 = self.projector(y)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()