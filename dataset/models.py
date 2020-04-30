import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name


    def save(self, path, epoch=0):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(self.state_dict(), 
                os.path.join(complete_path, 
                    "model-{}.pth".format(str(epoch).zfill(5))))


    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")
        

    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            model_files = glob.glob(complete_path+"/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        self.load_state_dict(torch.load(mf))


class MVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        super(MVCNN, self).__init__(name)
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                model = models.resnet18(pretrained=self.pretraining)
            elif self.cnn_name == 'resnet34':
                model = models.resnet34(pretrained=self.pretraining)
            elif self.cnn_name == 'resnet50':
                model = models.resnet50(pretrained=self.pretraining)
            self.net = nn.Sequential(*list(model.children())[:-1])
        else:
            if self.cnn_name == 'alexnet':
                self.net = models.alexnet(pretrained=self.pretraining).features
            elif self.cnn_name == 'vgg11':
                self.net = models.vgg11(pretrained=self.pretraining).features
            elif self.cnn_name == 'vgg16':
                self.net = models.vgg16(pretrained=self.pretraining).features

    def forward(self, x):
        y = self.net(x)
        return y.view(y.shape[0],-1)

class MVFC(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11', num_views=[]):
        super(MVFC, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.num_views = num_views
        l_views = len(num_views)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = nn.Sequential(
                    nn.Linear(l_views*512,4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, nclasses),
                )
            elif self.cnn_name == 'resnet34':
                self.net = nn.Linear(l_views*512,nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = nn.Linear(l_views*2048,nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6 * l_views, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, nclasses),
                )
            elif self.cnn_name == 'vgg11':
                self.net = nn.Sequential(
                    nn.Linear(512 * 7 * 7 * l_views, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, nclasses),
                )
            elif self.cnn_name == 'vgg16':
                self.net = nn.Sequential(
                    nn.Linear(512 * 7 * 7 * l_views, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, nclasses),
                )

    def forward(self, x):
        return self.net(x)

class MVFCG(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11', shape = []):
        super(MVFCG, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.shape = shape

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = nn.Sequential(
                    nn.Linear(shape[0],shape[1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(shape[1], shape[2]),
                )
            elif self.cnn_name == 'resnet34':
                self.net = nn.Linear(shape[0],shape[1])
            elif self.cnn_name == 'resnet50':
                self.net = nn.Linear(shape[0],shape[1])
        else:
            if self.cnn_name == 'alexnet':
                self.net = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(shape[0], shape[1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(shape[1], shape[2]),
                    nn.ReLU(inplace=True),
                    nn.Linear(shape[2], shape[3]),
                )
            elif self.cnn_name == 'vgg11':
                self.net = nn.Sequential(
                    nn.Linear(shape[0], shape[1]),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(shape[1], shape[2]),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(shape[2], shape[3]),
                )
            elif self.cnn_name == 'vgg16':
                self.net = nn.Sequential(
                    nn.Linear(shape[0], shape[1]),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(shape[1], shape[2]),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(shape[2], shape[3]),
                )

    def forward(self, x):
        return self.net(x)