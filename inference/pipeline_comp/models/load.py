import torch
import torch.nn as nn
import torchvision.models as models
import sys

sys.path.insert(0, "/hom2/deepti.rawat/anaconda3/envs/keshav/lib/python3.9/site-packages/ultralytics/")

class CustomClassifier(torch.nn.Module):
    def __init__(self, in_features, num_classes=3):
        super(CustomClassifier, self).__init__()
        self.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x

class CustomLayer(torch.nn.Module):
    def __init__(self, in_features, num_classes=3):
        super(CustomLayer, self).__init__()
        self.conv = nn.Conv2d(32, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.fc = torch.nn.Linear(32*5*5, num_classes)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.act = nn.ReLU()

    def forward(self, x):
        # x has shape 
        x = x.reshape(x.shape[0], 32, 28, 28)
        x = self.act(self.conv(x))
        x = self.avgpool(x)
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class CustomLayerVar(torch.nn.Module):
    def __init__(self, in_features, num_classes=3):
        super(CustomLayerVar, self).__init__()
        self.conv = nn.Conv2d(32, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.fc = torch.nn.Linear(32*256, 16)
        self.fc2 = torch.nn.Linear(16, num_classes)
        self.adaptive1d = nn.AdaptiveAvgPool1d(256)
        self.act = nn.ReLU()

    def forward(self, x):
        # x has shape (b, 32, h/8, w/8)
        x = x.reshape(1, 32, -1)
        x = self.adaptive1d(x).reshape(1, -1)
        # x = self.act(self.conv(x))
        # x = self.avgpool(x)
        # x = self.act(self.conv2(x))
        # x = self.act(self.conv3(x))
        # x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)
        x = self.act(self.fc(x))
        x = self.fc2(x)
        return x
    
def give_resnet_model(num_classes=3):
    resnet18 = models.resnet18(pretrained=True)
    fc_feat = resnet18.fc.in_features
    last_layer = CustomClassifier(fc_feat, num_classes=num_classes)
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])
    resnet18.add_module('fc', last_layer)
    model = resnet18
    return model

def give_yolo_model(weights="/ssd_scratch/cvit/keshav/model_ft.pt", num_classes=3):
    model = torch.load(weights)
    # model = model['model'].float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clf_layer = nn.Sequential(nn.Linear(49*32*16, 32), nn.ReLU(), nn.Linear(32, num_classes)).to(device)
    clf_layer = CustomLayer(49*32*16, num_classes).to(device)
    
    model['model'].model[-1].classifier = clf_layer #nn.Linear(49*32*16, num_classes).to(device)
    model['model'].model[-1].layer = 0
    unfreeze = ['classifier', 'cv4', '15']
    for name, parm in model['model'].model.named_parameters():
        print(name, parm.shape)
        for fl in unfreeze:
            if(fl not in name):
                parm.requires_grad=False
            else:
                parm.requires_grad=True
                break
    model = model['model'].float()
    return model