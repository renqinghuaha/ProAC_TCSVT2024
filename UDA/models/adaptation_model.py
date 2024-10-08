import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import numpy as np

from schedulers import get_scheduler
from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from models.deeplab import DeepLab
from models.deeplabv2plus import Deeplabplus
from models.discriminator import FCDiscriminator
from .utils import freeze_bn, GradReverse, cross_entropy2d
from models.loss import PixelContrastLoss

class CustomModel():
    def __init__(self, cfg, writer, logger):
        self.cfg = cfg
        self.writer = writer
        self.class_numbers = 19
        self.logger = logger
        cfg_model = cfg['model']
        self.cfg_model = cfg_model
        self.best_iou = -100
        self.iter = 0
        self.nets = []
        self.split_gpu = 0
        self.default_gpu = cfg['model']['default_gpu']
        self.valid_classes = cfg['training']['valid_classes']
        self.G_train = True

        bn = cfg_model['bn']

        self.optimizers = []
        self.schedulers = []

        self.BaseNet = DeepLab(
                            num_classes=19,
                            backbone=cfg_model['basenet']['version'],
                            output_stride=8,
                            bn=cfg_model['bn'],
                            freeze_bn=False,
                            )
        logger.info('the backbone is {}'.format(cfg_model['basenet']['version']))
        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=True)
        self.nets.extend([self.BaseNet])
        self.nets_DP = [self.BaseNet_DP]
        optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items()
                            if k != 'name'}
        self.BaseOpti = torch.optim.SGD([{'params': self.BaseNet.get_1x_lr_params(), 'lr': optimizer_params['lr']},
                                         {'params': self.BaseNet.get_10x_lr_params(), 'lr': optimizer_params['lr'] * 10}], **optimizer_params)
        self.optimizers.extend([self.BaseOpti])
        self.BaseSchedule = get_scheduler(self.BaseOpti, cfg['training']['lr_schedule'])
        self.schedulers.extend([self.BaseSchedule])
        # self.setup(cfg, writer, logger)
        default_gpu = cfg['model']['default_gpu']
        device = torch.device("cuda:{}".format(default_gpu) if torch.cuda.is_available() else 'cpu')
        self.pcl_loss = PixelContrastLoss(cfg).to(device)
        self.optimal_target = torch.load('optimal_target_19_256.pth')
        self.optimal_target = self.optimal_target.cuda()

    def forward(self, input):
        return self.BaseNet_DP(input)

    def setup(self, cfg, writer, logger):
        '''
        set optimizer and load pretrained model
        '''
        for net in self.nets:
            # name = net.__class__.__name__
            self.init_weights(cfg['model']['init'], logger, net)
            print("Initializition completed")
            if hasattr(net, '_load_pretrained_model') and cfg['model']['pretrained']:
                print("loading pretrained model for {}".format(net.__class__.__name__))
                net._load_pretrained_model()
        '''load pretrained model
        '''
        if cfg['training']['resume_flag']:
            self.load_nets(cfg, writer, logger)
        pass

    def step_contrast(self, source_x, source_label, target_x, target_label):
        self.BaseOpti.zero_grad()
        source_output = self.BaseNet_DP(source_x)
        source_outputUp = F.interpolate(source_output['out'], size=source_x.size()[2:], mode='bilinear', align_corners=True)
        loss_GTA = cross_entropy2d(input=source_outputUp, target=source_label.squeeze(1))

        target_output = self.BaseNet_DP(target_x)
        target_outputUp = F.interpolate(target_output['out'], size=target_x.size()[2:], mode='bilinear', align_corners=True)
        loss_CTS = cross_entropy2d(input=target_outputUp, target=target_label)

        all_feats = torch.cat([source_output['feat'], target_output['feat']], dim=0)
        all_labels = torch.cat([source_label.squeeze(1), target_label.squeeze(1)], dim=0)
        loss_feat = self.pcl_loss(all_feats, all_labels, self.optimal_target)
        loss = loss_GTA + loss_CTS + 0.1*loss_feat
        
        loss.backward()
        self.BaseOpti.step()
        return loss_GTA, loss_CTS, loss_feat

    def freeze_bn_apply(self):
        for net in self.nets:
            net.apply(freeze_bn)
        for net in self.nets_DP:
            net.apply(freeze_bn)

    def scheduler_step(self):
        # for net in self.nets:
        #     self.schedulers[net.__class__.__name__].step()
        for scheduler in self.schedulers:
            scheduler.step()
    
    def optimizer_zerograd(self):
        # for net in self.nets:
        #     self.optimizers[net.__class__.__name__].zero_grad()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    
    def optimizer_step(self):
        # for net in self.nets:
        #     self.optimizers[net.__class__.__name__].step()
        for opt in self.optimizers:
            opt.step()

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        # if torch.cuda.is_available():
        if whether_DP:
            net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
        return net
    
    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        if net == None:
            for net in self.nets:
                net.eval()
            for net in self.nets_DP:
                net.eval()
            if logger!=None:    
                logger.info("Successfully set the model eval mode") 
        else:
            net.eval()
            if logger!=None:    
                logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return

    def train(self, net=None, logger=None):
        if net==None:
            for net in self.nets:
                net.train()
            for net in self.nets_DP:
                net.train()
            # if logger!=None:    
            #     logger.info("Successfully set the model train mode") 
        else:
            net.train()
            # if logger!= None:
            #     logger.info(print("Successfully set {} train mode".format(net.__class__.__name__)))
        return

    def set_requires_grad(self, logger, net, requires_grad = False):
        """Set requires_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            net (BaseModel)       -- the network which will be operated on
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        for parameter in net.parameters():
            parameter.requires_grad = requires_grad
        # print("Successfully set {} requires_grad with {}".format(net.__class__.__name__, requires_grad))
        # return
        
    def set_requires_grad_layer(self, logger, net, layer_type='batchnorm', requires_grad=False):  
        '''    set specific type of layers whether needing grad
        '''

        # print('Warning: all the BatchNorm params are fixed!')
        # logger.info('Warning: all the BatchNorm params are fixed!')
        for net in self.nets:
            for _i in net.modules():
                if _i.__class__.__name__.lower().find(layer_type.lower()) != -1:
                    _i.weight.requires_grad = requires_grad
        return

    def init_weights(self, cfg, logger, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        init_type = cfg.get('init_type', init_type)
        init_gain = cfg.get('init_gain', init_gain)
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, SynchronizedBatchNorm2d) or classname.find('BatchNorm2d') != -1 \
                or isinstance(m, nn.GroupNorm):
                # or isinstance(m, InPlaceABN) or isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_() # BatchNorm Layer's weight is not a matrix; only normal distribution applies.


        print('initialize {} with {}'.format(init_type, net.__class__.__name__))
        logger.info('initialize {} with {}'.format(init_type, net.__class__.__name__))
        net.apply(init_func)  # apply the initialization function <init_func>
        pass

    def adaptive_load_nets(self, net, model_weight):
        model_dict = net.state_dict()
        pretrained_dict = {k : v for k, v in model_weight.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    def load_nets(self, cfg, writer, logger):    # load pretrained weights on the net
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            _k = -1
            for net in self.nets:
                name = net.__class__.__name__
                _k += 1
                if checkpoint.get(name) == None:
                    continue
                if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                    continue
                self.adaptive_load_nets(net, checkpoint[name]["model_state"])
                if cfg['training']['optimizer_resume']:
                    self.adaptive_load_nets(self.optimizers[_k], checkpoint[name]["optimizer_state"])
                    self.adaptive_load_nets(self.schedulers[_k], checkpoint[name]["scheduler_state"])
            self.iter = checkpoint["iter"]
            self.best_iou = checkpoint['best_iou']
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["iter"]
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(cfg['training']['resume']))

    def set_optimizer(self, optimizer):  #set optimizer to all nets
        pass

def grad_reverse(x):
    return GradReverse()(x)