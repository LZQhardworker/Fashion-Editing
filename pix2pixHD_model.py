import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from base_model import BaseModel
import networks
import cv2
import util.util as util
import torch.nn.functional as F
NC=20
def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], 20))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label

def generate_discrete_label(inputs, label_nc):
    pred_batch = []
    size = inputs.size()
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_map = []
    for p in pred_batch:
        p = p.view(1, 512, 320)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
                      
    return input_label,label_map

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        # Main Generator
        self.netG = networks.define_G(11, opt.output_nc, opt.ngf, opt.netG,
                                          opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                          opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        self.netP = networks.define_P(44, 20, opt.ngf, opt.netG,
                                           opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                           opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.netP.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/generate/parse.pth"))




        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            netB_input_nc = opt.output_nc * 2
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            #self.netB = networks.define_B(netB_input_nc, opt.output_nc, 32, 3, 3, opt.norm, gpu_ids=self.gpu_ids)        
            
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain                        
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)

            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
        
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionStyle=networks.StyleLoss(self.gpu_ids)
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real','D_fake')
            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())+list(self.netimage.parameters())+list(self.netcolor.parameters())\
                         +list(self.netlabel.parameters())+list(self.netsketch.parameters())+list(self.classfier.parameters())
                # params.extend(list(self.netimage.parameters()))
            
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
   
            # optimizer G + B                        
            #params = list(self.netG.parameters()) + list(self.netB.parameters())     
            #self.optimizer_GB = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
    
    def encode_input(self, inter_label_map_1, label_map, inter_label_map_2, real_image, label_map_ref, real_image_ref, infer=False):
        
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
            inter_label_1 = inter_label_map_1.data.cuda()
            inter_label_2 = inter_label_map_2.data.cuda() 
            input_label_ref = label_map_ref.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            inter_label_1 = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            inter_label_1 = inter_label_1.scatter_(1, inter_label_map_1.data.long().cuda(), 1.0)
            inter_label_2 = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            inter_label_2 = inter_label_2.scatter_(1, inter_label_map_2.data.long().cuda(), 1.0)
            input_label_ref = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label_ref = input_label_ref.scatter_(1, label_map_ref.data.long().cuda(), 1.0) 
            if self.opt.data_type == 16:
                input_label = input_label.half()
                inter_label_1 = inter_label_1.half()
                inter_label_2 = inter_label_2.half() 
                input_label_ref = input_label_ref.half()
 
        input_label = Variable(input_label, volatile=infer)
        inter_label_1 = Variable(inter_label_1, volatile=infer)
        inter_label_2 = Variable(inter_label_2, volatile=infer)
        input_label_ref = Variable(input_label_ref, volatile=infer)        
        real_image = Variable(real_image.data.cuda()) 
        real_image_ref = Variable(real_image_ref.data.cuda())
        
        return inter_label_1, input_label, inter_label_2, real_image, input_label_ref, real_image_ref

    def encode_input_test(self, label_map, real_image_ref, infer=False):

        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        input_label = Variable(input_label, volatile=infer)
        real_image_ref = Variable(real_image_ref.data.cuda())

        return input_label, real_image_ref

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)



    def inference(self, label, image_ref, edge, color, mask, noise):

        # Encode Inputs
        image_ref = Variable(image_ref)
        input_label, real_image_ref = self.encode_input_test(Variable(label), image_ref, infer=True)
        with torch.no_grad():
            masked_label = (1 - mask) * input_label
            edge = edge * mask
            color = color * mask
            noise = noise * mask
            image_ref = image_ref * (1 - mask)

            synthesized_label = self.netP(
                torch.cat([masked_label, mask + torch.zeros(input_label.shape).cuda(), color, edge], dim=1),
                torch.cat([edge, color], dim=1), noise)



            synthesized_label, input_plain =generate_discrete_label(synthesized_label, 20)

            M_complete = (input_plain > 0).cpu().numpy().astype(np.int)
            M_complete = torch.FloatTensor(M_complete).cuda() * (1 - mask) + torch.zeros(image_ref.shape).cuda()
            #
            fake_image = self.netG.forward(torch.cat([image_ref, M_complete, noise, color, edge], dim=1),
                                           torch.cat([edge, color, noise], dim=1), synthesized_label)
            label_color = generate_label_color(input_plain).squeeze()
            label_color = label_color.permute(1, 2, 0).cpu().numpy()
            cv2.imshow('a', label_color)
            cv2.waitKey(0)
            # print(label_color.shape)
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        #self.save_network(self.netB, 'B', which_epoch, self.gpu_ids)
      
    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label = inp
        return self.inference(label)

        
