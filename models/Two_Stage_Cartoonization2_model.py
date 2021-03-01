import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os
# from .gradient_loss1 import get_optics
# from .Guidance_loss2 import get_guidance
from .Flattening_Loss2 import Flattening_Loss2_Gause
from .Gause_Kernel import Gause_Kernel
# from .RGB2YUV import RGB2YUV 

class TWOSTAGECARTOONIZATION2Model(BaseModel):
    def name(self):
        return 'TWOSTAGECARTOONIZATION2Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--identity_A', type=float, default=20.0, help='weight for more stable abstraction')
            parser.add_argument('--identity_L', type=float, default=20.0, help='weight for more stable line drawing')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['conA','fla','GDA', 'G_A', 'conL', 'GDL', 'G_L', 'D_L']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names= ['real_A', 'RGA', 'RGL', 'cartoon_L', 'real_BS', 'real_BS2', 'real_BS3', 'real_BS4', 'real_BS5', 'real_B'] # train
        else:
            self.visual_names = ['RGA', 'RGL'] # test

        if self.isTrain and self.opt.identity_A > 0.0:
            self.loss_names.append('idt_A')
            self.visual_names.append('idt_A')
        if self.isTrain and self.opt.identity_L > 0.0:
            self.loss_names.append('idt_L')
            self.visual_names.append('idt_L')
            
        if self.isTrain:
            self.model_names = ['G_A', 'G_L', 'D_A', 'D_L']
        else:
            self.model_names = ['G_A', 'G_L']

        self.netG_A = networks.define_C(opt.input_nc, opt.output_nc, opt.ngf, opt.netGA, opt.norm, # 'resnet_9blocks_PG_Up'
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_L = networks.define_C(opt.input_nc, opt.output_nc, opt.ngf, opt.netGL, opt.norm, # 'resnet_6blocks'
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        # Discriminator
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netDA,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_L = networks.define_D(opt.output_nc, opt.ndf, opt.netDL,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.RGA_pool = ImagePool(opt.pool_size)
            self.RGL_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            
            # define Gause kernel
            es = 5.
            self.gause_radius = 5
            self.Kernel = Gause_Kernel(self.gause_radius, opt.batch_size, es).to(self.device)
            
            # initialize optimizers
            self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_L = torch.optim.Adam(self.netG_L.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_L = torch.optim.Adam(self.netD_L.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_L)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_L)



    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if self.isTrain:
            self.real_AE = input['AE'].to(self.device)
        if self.isTrain:
            self.real_B = input['B' if AtoB else 'A'].to(self.device) # when test, it needs to be annotated
        if self.isTrain:
            self.real_B2 = input['B2' if AtoB else 'A'].to(self.device) # when test, it needs to be annotated        
        if self.isTrain:
            self.real_B3 = input['B3' if AtoB else 'A'].to(self.device) # when test, it needs to be annotated
        if self.isTrain:
            self.real_B4 = input['B4' if AtoB else 'A'].to(self.device) # when test, it needs to be annotated
        if self.isTrain:
            self.real_B5 = input['B5' if AtoB else 'A'].to(self.device) # when test, it needs to be annotated

        if self.isTrain:
            self.real_BS = input['BS' if AtoB else 'A'].to(self.device) # when test, it needs to be annotated
        if self.isTrain:
            self.real_BS2 = input['BS2' if AtoB else 'A'].to(self.device) # when test, it needs to be annotated        
        if self.isTrain:
            self.real_BS3 = input['BS3' if AtoB else 'A'].to(self.device) # when test, it needs to be annotated
        if self.isTrain:
            self.real_BS4 = input['BS4' if AtoB else 'A'].to(self.device) # when twaszest, it needs to be annotated
        if self.isTrain:
            self.real_BS5 = input['BS5' if AtoB else 'A'].to(self.device) # when test, it needs to be annotated
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.RGA = self.netG_A(self.real_A)
        if self.isTrain: # idt
            self.RGL = self.netG_L(self.RGA)
            self.cartoon_L = self.netG_L(self.real_BS)
            if self.opt.identity_A:
                self.idt_A = self.netG_A(self.real_B)
            if self.opt.identity_L:
                self.idt_L = self.netG_L(self.idt_A)

    def backward_D_basic5(self, netD, real, real2, real3, real4, real5, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Real2
        pred_real2 = netD(real2)
        loss_D_real2 = self.criterionGAN(pred_real2, True)
        # Real3
        pred_real3 = netD(real3)
        loss_D_real3 = self.criterionGAN(pred_real3, True)
        # Real4
        pred_real4 = netD(real4)
        loss_D_real4 = self.criterionGAN(pred_real4, True)
        # Real5
        pred_real5 = netD(real5)
        loss_D_real5 = self.criterionGAN(pred_real5, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # # Combined loss * 2
        loss_D = (loss_D_real + loss_D_real2 + loss_D_real3 + loss_D_real4 + loss_D_real5) * 0.05 + loss_D_fake * 0.25
        loss_D.backward()
        return loss_D

    def backward_D_basic3(self, netD, real, fake, real_fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real_Fake
        pred_real_fake = netD(real_fake)
        loss_D_real_fake = self.criterionGAN(pred_real_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake + loss_D_real_fake) / 3.0
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        RGA = self.RGA_pool.query(self.RGA)
        self.loss_D_A = self.backward_D_basic5(self.netD_A, self.real_BS, self.real_BS2, self.real_BS3, self.real_BS4, self.real_BS5, RGA)

    def backward_D_L(self):
        RGL = self.RGL_pool.query(self.RGL)
        self.loss_D_L = self.backward_D_basic3(self.netD_L, self.real_B, RGL, self.real_BS)

    def backward_G_first(self):
        mask = -0.5 * (self.real_AE - 1.) # 1 represents important edge area while 0 represents blank.
        mask = torch.cat((mask, mask, mask), dim=1)
        self.loss_conA = torch.mean((1-mask*0.1).mul(torch.pow(self.real_A - self.RGA, 2))) * 10
        if self.opt.identity_A > 0.0:
            self.loss_idt_A = self.criterionL1(self.idt_A, self.real_BS)*20
        else:
            self.loss_idt_A = torch.tensor([0.]).to(self.device)
        self.loss_fla =  torch.tensor([0.]).to(self.device)
        self.loss_GDA =  torch.tensor([0.]).to(self.device)
        self.loss_G_A = self.loss_conA + self.loss_idt_A + self.loss_fla + self.loss_GDA

        self.loss_conL = self.criterionL1(self.cartoon_L, self.real_B) * 50
        self.loss_idt_L = torch.tensor([0.]).to(self.device)
        self.loss_GDL = torch.tensor([0.]).to(self.device)
        self.loss_G_L = self.loss_conL + self.loss_idt_L + self.loss_GDL
        
        self.loss_D_A =  torch.tensor([0.]).to(self.device)
        self.loss_D_L = torch.tensor([0.]).to(self.device)

    def backward_G_second(self):
        mask = -0.5 * (self.real_AE - 1.) # 1 represents important edge area while 0 represents blank.
        mask = torch.cat((mask, mask, mask), dim=1)
        self.loss_conA = torch.mean((1-mask*0.1).mul(torch.pow(self.real_A - self.RGA, 2))) * 10
        if self.opt.identity_A > 0.0:
            self.loss_idt_A = self.criterionL1(self.idt_A, self.real_BS)*20
        else:
            self.loss_idt_A = torch.tensor([0.]).to(self.device)
        self.loss_fla = Flattening_Loss2_Gause(self.device, self.real_A, self.RGA, self.gause_radius, 0.2, 1, self.Kernel)*1.5
        self.loss_GDA = self.criterionGAN(self.netD_A(self.RGA), True)*2
        self.loss_G_A = self.loss_conA + self.loss_idt_A + self.loss_fla + self.loss_GDA

        self.loss_conL = self.criterionL1(self.cartoon_L, self.real_B) * 50 # Hayao50
        self.loss_idt_L = torch.tensor([0.]).to(self.device)
        self.loss_GDL = torch.tensor([0.]).to(self.device)
        self.loss_G_L = self.loss_conL + self.loss_idt_L + self.loss_GDL
        
        self.loss_D_L = torch.tensor([0.]).to(self.device)

    def backward_G_third(self):
        mask = -0.5 * (self.real_AE - 1.) # 1 represents important edge area while 0 represents blank.
        mask = torch.cat((mask, mask, mask), dim=1)
        self.loss_conA = torch.mean((1-mask*0.1).mul(torch.pow(self.real_A - self.RGA, 2))) * 10
        if self.opt.identity_A > 0.0:
            self.loss_idt_A = self.criterionL1(self.idt_A, self.real_BS)*20
        else:
            self.loss_idt_A = torch.tensor([0.]).to(self.device)
        self.loss_fla = torch.tensor([0.]).to(self.device)
        self.loss_GDA = self.criterionGAN(self.netD_A(self.RGA), True)*2
        self.loss_G_A = self.loss_conA + self.loss_idt_A + self.loss_fla + self.loss_GDA

        self.loss_conL = self.criterionL1(self.cartoon_L, self.real_B) * 50 # Hayao50
        if self.opt.identity_L > 0.0:
            self.loss_idt_L = self.criterionL1(self.idt_L, self.real_B) * 20
        else:
            self.loss_idt_L = torch.tensor([0.]).to(self.device)
        self.loss_GDL = self.criterionGAN(self.netD_L(self.RGL), True)
        self.loss_G_L = self.loss_conL + self.loss_idt_L + self.loss_GDL
        
    def backward_G_last(self):
        self.loss_conA = torch.tensor([0.]).to(self.device)
        self.loss_idt_A = torch.tensor([0.]).to(self.device)
        self.loss_fla = torch.tensor([0.]).to(self.device)
        self.loss_GDA = torch.tensor([0.]).to(self.device)
        self.loss_G_A = self.loss_conA + self.loss_idt_A + self.loss_fla + self.loss_GDA

        self.loss_D_A =  torch.tensor([0.]).to(self.device)
        
        self.loss_conL = self.criterionL1(self.cartoon_L, self.real_B) * 50 # Hayao50
        if self.opt.identity_L > 0.0:
            self.loss_idt_L = self.criterionL1(self.idt_L, self.real_B) * 20
        else:
            self.loss_idt_L = torch.tensor([0.]).to(self.device)
        self.loss_GDL = self.criterionGAN(self.netD_L(self.RGL), True)
        self.loss_G_L = self.loss_conL + self.loss_idt_L + self.loss_GDL

    def optimize_parameters_first(self):
        self.forward()
        self.backward_G_first()
        self.set_requires_grad(self.netD_A, False)
        self.set_requires_grad(self.netD_L, False)
        self.set_requires_grad(self.netG_A, False)
        self.optimizer_G_L.zero_grad() 
        self.loss_G_L.backward()
        self.optimizer_G_L.step()

        self.set_requires_grad(self.netG_A, True)
        self.optimizer_G_A.zero_grad()
        self.loss_G_A.backward()
        self.optimizer_G_A.step()       

    def optimize_parameters_second(self):
        self.forward()
        self.backward_G_second()
        self.set_requires_grad(self.netD_A, False)
        self.set_requires_grad(self.netD_L, False)
        # G_L
        self.set_requires_grad(self.netG_A, False)
        self.optimizer_G_L.zero_grad() 
        self.loss_G_L.backward(retain_graph=True) 
        self.optimizer_G_L.step()

        # G_A
        self.set_requires_grad(self.netG_A, True)
        self.optimizer_G_A.zero_grad()
        self.loss_G_A.backward()
        self.optimizer_G_A.step()     

        # D_A
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

    def optimize_parameters_third(self):
        self.forward()
        self.backward_G_third()
        self.set_requires_grad(self.netD_A, False)
        self.set_requires_grad(self.netD_L, False)
        # G_L
        self.set_requires_grad(self.netG_A, False)
        self.optimizer_G_L.zero_grad() 
        self.loss_G_L.backward(retain_graph=True) 
        self.optimizer_G_L.step()

        # G_A
        self.set_requires_grad(self.netG_A, True)
        self.optimizer_G_A.zero_grad()
        self.loss_G_A.backward()
        self.optimizer_G_A.step()     

        # D_A
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()        

        # D_L
        self.set_requires_grad(self.netD_L, True)
        self.optimizer_D_L.zero_grad()
        self.backward_D_L()
        self.optimizer_D_L.step()  
        
    def optimize_parameters_last(self):
        self.forward()
        self.backward_G_last()
        self.set_requires_grad(self.netD_A, False)
        self.set_requires_grad(self.netD_L, False)
        # G_L
        self.set_requires_grad(self.netG_A, False)
        self.optimizer_G_L.zero_grad() 
        self.loss_G_L.backward(retain_graph=True)
        self.optimizer_G_L.step()
        
        # D_L
        self.set_requires_grad(self.netD_L, True)
        self.optimizer_D_L.zero_grad()
        self.backward_D_L()
        self.optimizer_D_L.step()        
