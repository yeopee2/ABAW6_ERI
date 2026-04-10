from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
#from torchinfo import summary
import numpy as np

#from torchinfo import summary
import networks.resnet as ResNet
#from utils.builder import get_builder

# +
N_IDENTITY = 8631 
class DAN50(nn.Module):
    def __init__(self, pretrained=True, num_head=4, num_class=8):
        super(DAN50, self).__init__()
        
        self.num_head = num_head
        
        include_top = True 
        #resnet = ResNet.resnet50(pretrained_checkpoint_path="./models/resnet50_scratch_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
        resnet = ResNet.resnet50(pretrained_checkpoint_path="./models/resnet50_ft_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
        self.num_class = num_class
        #resnet = SeNet.senet50(pretrained_checkpoint_path="./models/senet50_scratch_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        #print(self.features)

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        #self.fc1 = 
 
        self.model = DANfor50(pretrained = pretrained, num_class = self.num_class, num_head = self.num_head)
        if pretrained == "Yes" :
            checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict = False)
        self.dan=nn.Sequential(*list(self.model.children()))
        #print('dan',self.dan)

    def forward(self, x):
        x=self.features(x)
        # print(np.shape(x))
        x=self.conv1x1_1(x) 
        # print(np.shape(x))
        x=self.conv1x1_2(x)
        #print(np.shape(x))
        out,x,heads=self.model(x)
       
        return out, x, heads
class QuantDAN50(nn.Module):
    def __init__(self, pretrained=True, num_head=4, num_class=8):
        super(QuantDAN50, self).__init__()
        
        self.num_head = num_head
        
        include_top = True 
        self.quant = torch.quantization.QuantStub()
        #resnet = ResNet.resnet50(pretrained_checkpoint_path="./models/resnet50_scratch_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
        resnet = ResNet.resnet50(pretrained_checkpoint_path="./models/resnet50_ft_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
        self.num_class = num_class

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        #print(self.features)

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        #self.fc1 = 
 
        self.model = DANfor50(pretrained = pretrained, num_class = self.num_class, num_head = self.num_head)
        if pretrained == "Yes" :
            checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict = False)
        self.dan=nn.Sequential(*list(self.model.children()))
        self.dequant = torch.quantization.DeQuantStub()
        #print('dan',self.dan)

    def forward(self, x):

        x=self.features(x)

        # print(np.shape(x))
        
        x=self.conv1x1_1(x) 
        # print(np.shape(x))
        x=self.conv1x1_2(x)
        #print(np.shape(x))

        out,x,heads=self.model(x)
        
        #out = self.dequant(out)
        #heads = self.dequant(heads)
        return out, x, heads
    
class resnetmtl(nn.Module):
    def __init__(self, num_head = 4,num_class=20, pretrained=True):
        super(resnetmtl, self).__init__()
        

        self.num_head=num_head
    
        self.avp2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #print(self.resnetfc)
        self.DAN = DAN(num_head = num_head,num_class=8)
        if pretrained:
            checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
            self.DAN.load_state_dict(checkpoint['model_state_dict'],strict=False)
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        for i in range(num_head):
            setattr(self,"cat_head1%d" %i, CrossAttentionHead())

        self.fc1 = nn.Linear(512,8)
        #self.softmax1 = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm1d(8)

        self.fc2 = nn.Linear(512,12)
        self.sigmoid = nn.Sigmoid()
        
        self.fc3 = nn.Linear(512,2)
        self.tanh = nn.Tanh()
        
 
        self.fc4 = nn.Linear(14,8)
        self.bn4 = nn.BatchNorm1d(8)
    def forward(self, x):
        fer,feat, heads = self.DAN(x)
        #print(np.shape(x))

        x2=self.avp2d(feat)
        x2= x2.squeeze()
        au = self.fc2(fer)
        au = self.sigmoid(au)

    

        fer = self.fc1(fer)
        fer = self.bn1(fer)


        va = self.fc3(x2)
        va = self.tanh(va)
        


        return fer,au,va, feat,heads
class resnetmtl2(nn.Module):
    def __init__(self, num_head = 4,num_class=8, pretrained=True):
        super(resnetmtl2, self).__init__()
        

        self.num_head=num_head
    
        self.avp2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #print(self.resnetfc)
        self.DAN = DAN(num_head = 9,num_class=8)
        if pretrained:
            checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
            self.DAN.load_state_dict(checkpoint['model_state_dict'],strict=False)
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        self.headbn1 = nn.BatchNorm1d(512)
        self.encoder = nn.Sequential(
            nn.Linear(512*3,512),
            nn.BatchNorm1d(512),
        )
        self.fc1 = nn.Linear(512,8)
        #self.softmax1 = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm1d(8)

        self.fc2 = nn.Linear(512,12)
        self.sigmoid = nn.Sigmoid()
        
        self.fc3 = nn.Linear(512,2)
        self.tanh = nn.Tanh()
        
 
        self.fc4 = nn.Linear(14,8)
        self.bn4 = nn.BatchNorm1d(8)
    def forward(self, x):
        fer,feat, heads = self.DAN(x)
        #print(np.shape(x))
        heads1 = []
        for i in range(self.num_head):
            heads1.append(getattr(self,"cat_head%d" %i)(feat))
        
        heads1 = torch.stack(heads1).permute([1,0,2])
        if heads1.size(1)>1:
            heads1 = F.log_softmax(heads1,dim=1)

        x2=self.avp2d(feat)
        x2= x2.squeeze()
        au = heads1.sum(dim=1)
        va = self.headbn1(x2)
        
        au = self.fc2(au)
        au = self.sigmoid(au)

    

        fer = self.fc1(fer)
        fer = self.bn1(fer)


        va = self.fc3(va)
        va = self.tanh(va)
        


        return fer,au,va, feat,heads,heads1
class resnetmtl3(nn.Module):
    def __init__(self, num_head = 4,num_class=20, pretrained=True):
        super(resnetmtl3, self).__init__()
        

        self.num_head=num_head
    
        self.avp2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #print(self.resnetfc)
        self.DAN = DAN(num_head = 9,num_class=8)
        if pretrained:
            checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
            self.DAN.load_state_dict(checkpoint['model_state_dict'],strict=False)
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        for i in range(num_head):
            setattr(self,"cat_head1%d" %i, CrossAttentionHead())
        self.headbn1 = nn.BatchNorm1d(512)
        self.headbn2 = nn.BatchNorm1d(512)
        self.headbn3 = nn.BatchNorm1d(512)
        self.encoder = nn.Sequential(
            nn.Linear(512*3,512),

            nn.BatchNorm1d(512),
        )
        self.fc1 = nn.Linear(512,8)

        self.bn1 = nn.BatchNorm1d(8)

        self.fc2 = nn.Linear(512,12)
        self.sigmoid = nn.Sigmoid()
        
        self.fc3 = nn.Linear(512,2)
        self.tanh = nn.Tanh()
        
 
        self.fc4 = nn.Linear(14,8)
        self.bn4 = nn.BatchNorm1d(8)
    def forward(self, x):
        fer,feat, heads = self.DAN(x)
        
        heads2 = []
        for i in range(self.num_head):
            heads2.append(getattr(self,"cat_head%d" %i)(feat))
        
        heads2 = torch.stack(heads2).permute([1,0,2])

        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)
        
        heads1 = []
        for i in range(self.num_head):
            heads1.append(getattr(self,"cat_head1%d" %i)(feat))
        heads1 = torch.stack(heads1).permute([1,0,2])
        if heads1.size(1)>1:
            heads1 = F.log_softmax(heads1,dim=1)
        
        fer = heads.sum(dim=1)
        au = heads2.sum(dim=1)
        va = heads1.sum(dim=1)
       
        all_feat = torch.cat((au,va,fer),dim=1)
        all_feat = self.encoder(all_feat)


        au = self.fc2(all_feat)
        au = self.sigmoid(au)

    
        fer = self.fc1(all_feat)
        fer = self.bn1(fer)

        va = self.fc3(all_feat)
        va = self.tanh(va)
        


        return fer,au,va, feat,heads,heads1,heads2
class resnetmtl4(nn.Module):
    def __init__(self, num_head = 9,num_class=20, pretrained=True):
        super(resnetmtl4, self).__init__()
        self.num_head=num_head
    
        self.avp2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #print(self.resnetfc)
        self.DAN = DAN(num_head = 9,num_class=8)
        if pretrained:
            checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
            self.DAN.load_state_dict(checkpoint['model_state_dict'],strict=False)
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())

        self.encoder = nn.Sequential(
            nn.Linear(512*2,512),
            nn.BatchNorm1d(512),
        )
        self.fc1 = nn.Linear(512*2,8)
        #self.softmax1 = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm1d(8)

        self.fc2 = nn.Linear(512*2,12)
        self.sigmoid = nn.Sigmoid()
        
        self.fc3 = nn.Linear(512*2,2)
        self.tanh = nn.Tanh()
        
 
    def forward(self, x):
        fer,feat, heads = self.DAN(x)
        #print(np.shape(x))
        heads2 = []
        for i in range(self.num_head):
            heads2.append(getattr(self,"cat_head%d" %i)(feat))
        
        heads2 = torch.stack(heads2).permute([1,0,2])
        if heads2.size(1)>1:
            heads2 = F.log_softmax(heads2,dim=1)

        fer = heads.sum(dim=1)
        au = heads2.sum(dim=1)
        va = self.avp2d(feat)
        va= va.squeeze(3)
        va= va.squeeze(2)
        
        all_feat = torch.cat((au,fer),dim=1)
        all_feat = self.encoder(all_feat)


        au = torch.cat((au,all_feat),dim=1)
        au = self.fc2(au)
        au = self.sigmoid(au)

    
        fer = torch.cat((fer,all_feat),dim=1)
        fer = self.fc1(fer)
        fer = self.bn1(fer)

        va = torch.cat((va,all_feat),dim=1)
        va = self.fc3(va)
        va = self.tanh(va)
        


        return fer,au,va, feat, heads, heads2
class DAN(nn.Module):
    def __init__(self, num_class=8,num_head=4, pretrained=True):
        super(DAN, self).__init__()
        
        resnet = models.resnet18(pretrained)
        
        if pretrained:
            checkpoint = torch.load('./models/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())

        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

        self.fc2 = nn.Linear(num_class,12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)

        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)
            
        out = heads.sum(dim=1)

        #out = self.fc(out)
        #out = self.bn(out)
   
        return out, x, heads
class DANfor50(nn.Module):
    def __init__(self, num_class=8,num_head=4, pretrained=True):
        super(DANfor50, self).__init__()
        
        resnet = models.resnet18(pretrained)
        
        if pretrained:
            checkpoint = torch.load('./models/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())

        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

        self.fc2 = nn.Linear(num_class,12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = self.features(x)


        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
       # if heads.size(1)>1:
        heads = F.log_softmax(heads,dim=1)
            
        out = heads.sum(dim=1)

        out = self.fc(out)
        out = self.bn(out)
   
        return out, x, heads
class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca


# -

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()


    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1,keepdim=True) 
        out = x*y
        
        return out 

class ChannelAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )


    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0),-1)
        y = self.attention(sa)
        out = sa * y
        
        return out
class mobile_vit2(nn.Module) :
    def __init__(self, task) :
        super(mobile_vit2, self).__init__() 
        print("Loading MobileViT v2 !!")
        self.task = task
        opts = './mobilevitv2/config/classification/finetune_in21k_to_1k/mobilevit_v2.yaml'
        imagenet_pretrained_model_path = './mobilevitv2/checkpoints/mobilevitv2-2.0.pt'
        dev_id = getattr(opts, "dev.device_id", None)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(dev_id, device)
        if dev_id is None:
            model_state = torch.load(imagenet_pretrained_model_path, map_location=device)

        self.model = get_model(opts)

        model_state = torch.load(imagenet_pretrained_model_path)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state, strict = False)
        else:
            self.model.load_state_dict(model_state, strict = False)

        self.model = nn.Sequential(*(list(self.model.children())))
        self.model = self.model[:-1]
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(1024, 6) 

        print("Done !")
        


    def forward(self, x) :

        x = self.model(x)
        if self.task == 0 :
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x
