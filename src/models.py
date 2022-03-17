import torch
import torch.nn as nn
from torchvision.models import resnet
import os.path as osp
import numpy as np

class LeNet(nn.Module):
    def __init__(self, channel=3, hidden=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        fc_in = out.view(out.size(0), -1)
        fc_out = self.fc(fc_in)
        return fc_out, fc_in

def weights_init(m):
    torch.manual_seed(8)
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


class ResNet18(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        base = resnet.resnet18(pretrained=self.pretrained)
        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool)
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, self.num_classes, bias=True)
        for name, module in self.named_modules():
            if hasattr(module, 'relu'):
                module.relu = nn.Sigmoid()

    def forward(self, x):
        h = self.in_block(x)
        h = self.encoder1(h)
        h = self.encoder2(h)
        h = self.encoder3(h)
        h = self.encoder4(h)
        fc_in = torch.flatten(self.avgpool(h), 1)
        fc_out = self.fc(fc_in)
        return fc_out, fc_in

from src.VariationalBottleneck import VariationalBottleneck

class MLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        fc_in = self.relu(x)
        fc_out = self.l3(fc_in)
        return fc_out, fc_in

class VBMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super(VBMLP, self).__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.VB = VariationalBottleneck((width,))
        self.l3 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        fc_in = self.VB(x)
        fc_out = self.l3(fc_in)
        return fc_out, fc_in

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()

def get_model(channel, hidden, num_classes, args):

    if args.model == "lenet":
        model = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
    elif args.model == "resnet":
        model = ResNet18(num_classes=num_classes)
    elif args.model == "mlp":
        model = MLP(num_classes=num_classes)
    elif args.model == "vbmlp":
        model = VBMLP(num_classes=num_classes)
    else:
        NotImplementedError("undefined model")

    if args.load_model:
        print("load model ... ")
        checkpoint = torch.load(osp.join('pretrained', "{}_{}.pth.tar".format(args.model, args.dataset)))
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


if __name__ == '__main__':
    net = LeNet()
    x = torch.rand((4,3,224,224))
    y, _ = net(x)
    dummy_loss = torch.mean(y)
    for i in torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True):
        print(i.shape)

