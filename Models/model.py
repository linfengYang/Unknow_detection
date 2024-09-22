import numpy as np
from torch import nn
from Models.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Models.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec,attention_ROPE
import torch
import torch.nn.functional as F
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


def model_factory(config):
    if config['Net_Type'][0] == 'T':
        model = Transformer(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == '1D-2D':
        model = Image_Ts_combine(config, num_classes=config['num_labels'])                                      #todo
    else:
        model = ConvTran(config, num_classes=config['num_labels'])
    return model

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(
            nn.Linear(channel_size, emb_size),
            nn.LayerNorm(emb_size, eps=1e-5)
        )

        if self.Fix_pos_encode == 'Sin':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        if self.Rel_pos_encode == 'Scalar':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x_src = self.embed_layer(x.permute(0, 2, 1))
        if self.Fix_pos_encode != 'None':
            x_src = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        # out = out.permute(1, 0, 2)
        # out = self.out(out[-1])

        return out

class Image_Ts_combine1(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*8, kernel_size=[1, 16], padding='same'),
                                         nn.BatchNorm2d(emb_size*8),
                                         nn.GELU())
        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*8, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)      


        #############################第二部分#################################
        self.conv2=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),  
                nn.GELU(),
                nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),  
                nn.GELU(),
                nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),  
                nn.GELU(),
                nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(256*169,emb_size)                     #
        )

        self.new_out=nn.Linear(2*emb_size,num_classes)
        

    def forward(self, x, x2):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        #最后一层，等待融合
        # out = self.out(out)
        # return out
        x2=self.conv2(x2)
        new_x=torch.cat((out,x2),dim=1)
        new_x=self.flatten(new_x)
        new_out=self.new_out(new_x)
        return new_out
class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
   
class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        '''
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size*4),
                                         nn.GELU())
        '''
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*8, kernel_size=[1, 16], padding='same'),
                                         nn.BatchNorm2d(emb_size*8),
                                         nn.GELU())
        '''
        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())
        '''
        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*8, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out
'''
def ELA(x,channel,kernel_size):
    b,c,h,w=x.size()
    pad=kernel_size//2
    conv=nn.Conv1d(channel,channel,kernel_size=kernel_size,
                   padding=pad,groups=channel,bias=False)
    gn=nn.GroupNorm(16,channel)
    sigmoid=nn.Sigmoid()
    
    x_h=torch.mean(x,dim=3,keepdim=True).view(b,c,h)
    x_w=torch.mean(x,dim=2,keepdim=True).view(b,c,w)
    x_h=sigmoid(gn(conv(x_h))).view(b,c,h,1)
    x_w=sigmoid(gn(conv(x_w))).view(b,c,1,w)

    return x*x_h*x_w
'''
class ELA(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(ELA, self).__init__()
        self.pad=kernel_size//2
        self.conv=nn.Conv1d(channel,channel,kernel_size=kernel_size,
                   padding=self.pad,groups=channel,bias=False)
        self.gn=nn.GroupNorm(16,channel)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        b,c,h,w=x.size()
        x_h=torch.mean(x,dim=3,keepdim=True).view(b,c,h)
        x_w=torch.mean(x,dim=2,keepdim=True).view(b,c,w)
        x_h=self.sigmoid(self.gn(self.conv(x_h))).view(b,c,h,1)
        x_w=self.sigmoid(self.gn(self.conv(x_w))).view(b,c,1,w)

        return x*x_h*x_w

####################################resnet模型部分####################################################
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, is_use_CA=True):
        super(ResBlock, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        '''
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        '''
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(outchannel),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        # self.CA=CoordAtt(outchannel,outchannel)
        self.CA=CoordAtt(outchannel,outchannel)
        # self.CA = ELA(outchannel)
        self.is_CA=is_use_CA
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        xx=self.shortcut(x)
        if self.is_CA:
            out=self.CA(out)
            # xx=self.CA(xx)
            pass
        # out = out + self.shortcut(x)
            
        out = out+xx
        out = F.gelu(out)
        
        return out
    

class Image_Ts_combine2(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        '''
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
                                        nn.GELU(),
                                        nn.BatchNorm2d(emb_size*4))
        self.embed_layer1 = nn.Sequential(nn.Conv2d(emb_size*8, emb_size*4, kernel_size=[1, 8], padding='same'),
                                        nn.GELU(),      
                                        nn.BatchNorm2d(emb_size*4))
        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                        nn.GELU(),
                                        nn.BatchNorm2d(emb_size))
        '''
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*8, kernel_size=[1, 16], padding='same'),
                                        nn.BatchNorm2d(emb_size*8),
                                        nn.GELU())
        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*8, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                        nn.BatchNorm2d(emb_size),
                                        nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'attention_ROPE':
            self.attention_layer = attention_ROPE(emb_size,num_heads,seq_len,config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)      


        #############################第二部分#################################
        self.conv2=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  
                nn.GELU(),
                nn.BatchNorm2d(64),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1=nn.Sequential(*[ResBlock(64, 128, 2),ResBlock(128, 256, 2),ResBlock(256,512,2)])
        self.Layer2=nn.Linear(2048,emb_size)

        # self.Layer2=nn.Sequential(
        #     nn.Linear(512,128),
        #     nn.GELU(),
        #     nn.Dropout(0.02),
        #     nn.Linear(512,emb_size),
        #     nn.GELU(),
        #     nn.Dropout(0.02)
        # )    

        self.attention=CoordAtt(512,512)
        #self.attention=ELA(512)
        self.new_out=nn.Linear(2*emb_size,num_classes)
        '''
        nn.Conv2d(64, 128, kernel_size=4, stride=1),  
            nn.GELU(),
            nn.BatchNorm2d(128),
        nn.Conv2d(128, 256, kernel_size=3, stride=2),  
            nn.GELU(),
            nn.BatchNorm2d(256),
        nn.Flatten(),
        nn.Linear(256*169,emb_size)                     #
        '''
        self.new_out=nn.Linear(2*emb_size,num_classes)
        # self.second_mode=model_architecture(num_classes)

    def forward(self, x, x2):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        # x_src = self.embed_layer1(x)  
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        #最后一层，等待融合
        # out = self.out(out)
        # return out
        
        x2=self.conv2(x2)
        x2=self.layer1(x2)

        x2=x2+self.attention(x2)
        x2=F.avg_pool2d(x2,4)
        x2=self.flatten(x2)
        x2=self.Layer2(x2)

        # x2=self.second_mode(x2)
        new_x=torch.cat((out,x2),dim=1)
        new_x=self.flatten(new_x)
        new_out=self.new_out(new_x)
        return new_out


class Image_Ts_combine(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        '''
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
                                        nn.GELU(),
                                        nn.BatchNorm2d(emb_size*4))
        self.embed_layer1 = nn.Sequential(nn.Conv2d(emb_size*8, emb_size*4, kernel_size=[1, 8], padding='same'),
                                        nn.GELU(),      
                                        nn.BatchNorm2d(emb_size*4))
        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                        nn.GELU(),
                                        nn.BatchNorm2d(emb_size))
        '''
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*8, kernel_size=[1, 16], padding='same'),
                                        nn.BatchNorm2d(emb_size*8),
                                        nn.GELU())
        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*8, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                        nn.BatchNorm2d(emb_size),
                                        nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'attention_ROPE':
            self.attention_layer = attention_ROPE(emb_size,num_heads,seq_len,config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)      


        #############################第二部分#################################
        self.conv2=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  
                nn.GELU(),
                nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),  
            #     nn.GELU(),
            #     nn.BatchNorm2d(128),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1=nn.Sequential(*[ResBlock(64, 128, 2),ResBlock(128, 256, 2),ResBlock(256,512,2)])
        # self.layer1=nn.Sequential(*[ResBlock(128, 256, 2),ResBlock(256,512,2)])
        # self.Layer2=nn.Linear(512,emb_size)

        self.Layer2=nn.Sequential(
        #     nn.Linear(512,128),
        #     nn.GELU(),
        #     nn.Dropout(0.02),
            nn.Linear(512,emb_size),
            nn.GELU(),
            nn.Dropout(0.02)
        )    

        self.attention=CoordAtt(512,512)
        # self.attention=ELA(512)
        self.out=nn.Linear(emb_size,num_classes)
        '''
        nn.Conv2d(64, 128, kernel_size=4, stride=1),  
            nn.GELU(),
            nn.BatchNorm2d(128),
        nn.Conv2d(128, 256, kernel_size=3, stride=2),  
            nn.GELU(),
            nn.BatchNorm2d(256),
        nn.Flatten(),
        nn.Linear(256*169,emb_size)                     #
        '''
        self.new_out=nn.Linear(2*emb_size,num_classes)
        # self.second_mode=model_architecture(num_classes)

    def forward(self, x, x2,is_train=True):
        
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        # x_src = self.embed_layer1(x)  
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        
        #最后一层，等待融合
        # out = self.out(out)
        # return out
        
        x2=self.conv2(x2)
        x2=self.layer1(x2)

        x2=x2+self.attention(x2)
        x2=F.avg_pool2d(x2,4)
        x2=self.flatten(x2)
        x2=self.Layer2(x2)

        x2=self.flatten(x2)
        # x2=self.out(x2)
        # return x2
        # x2=self.second_mode(x2)
        
        new_x=torch.cat((out,x2),dim=1)
        new_out=self.flatten(new_x)
        if is_train:
            new_out=self.new_out(new_out)
        return new_out
        

