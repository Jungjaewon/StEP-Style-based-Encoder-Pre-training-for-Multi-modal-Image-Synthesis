import time
import datetime
import torch
import torch.nn as nn
import os.path as osp
import torch.backends.cudnn as cudnn
import glob
import os
import wandb

from torchvision import models as models
from torchvision.utils import save_image
from model import Discriminator
from model import UNet

from data_loader import get_data_loader


vgg_activation = dict()


def get_activation(name):
    def hook(model, input, output):
        vgg_activation[name] = output.detach()

    return hook


class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""

        assert torch.cuda.is_available()

        self.config = config
        self.wandb = config['TRAINING_CONFIG']['WANDB'] == 'True'
        self.seed = config['TRAINING_CONFIG']['SEED']
        self.num_style_feat = config['TRAINING_CONFIG']['NUM_STYLE_FEAT']
        self.percep = config['TRAINING_CONFIG']['PERCEP'] == 'True'
        if not self.percep:
            print(f'perceptual loss is not used')

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # https://hoya012.github.io/blog/reproducible_pytorch/

        self.train_data_loader = get_data_loader(config, 'train')
        self.val_data_loader = get_data_loader(config, 'val')
        self.test_data_loader = get_data_loader(config, 'test')

        self.E_model = config['TRAINING_CONFIG']['E_MODEL']
        self.E = None
        self.G = None
        self.D = None

        self.e_optimizer = None
        self.g_optimizer = None
        self.d_optimizer = None

        self.e_scheduler = None
        self.g_scheduler = None
        self.d_scheduler = None

        self.triplet_loss = nn.TripletMarginLoss(margin=0.2, p=2)
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        self.img_size = config['MODEL_CONFIG']['IMG_SIZE']

        assert self.img_size in [256]
        self.epoch1         = config['TRAINING_CONFIG']['EPOCH_1']
        self.epoch2         = config['TRAINING_CONFIG']['EPOCH_2']
        self.epoch3         = config['TRAINING_CONFIG']['EPOCH_3']
        self.batch_size     = config['TRAINING_CONFIG']['BATCH_SIZE']

        self.e_lr          = float(config['TRAINING_CONFIG']['E_LR'])
        self.g_lr          = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr          = float(config['TRAINING_CONFIG']['D_LR'])

        self.lambda_e_triplet = config['TRAINING_CONFIG']['LAMBDA_E_TRI']
        self.lambda_e_regular = config['TRAINING_CONFIG']['LAMBDA_E_REG']

        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_g_recon = config['TRAINING_CONFIG']['LAMBDA_G_RECON']
        self.lambda_g_percep = config['TRAINING_CONFIG']['LAMBDA_G_PERCEP']

        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']

        self.mse_loss = nn.MSELoss()
        self.gan_loss = config['TRAINING_CONFIG']['GAN_LOSS']
        assert self.gan_loss in ['lsgan']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        self.g_spec = config['TRAINING_CONFIG']['G_SPEC'] == 'True'
        self.d_spec = config['TRAINING_CONFIG']['D_SPEC'] == 'True'

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.gpu = torch.device(f'cuda:{self.gpu}')
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD'] == 'True'

        if self.percep:
            # vgg activation
            self.vgg = models.vgg19_bn(pretrained=True).eval().to(self.gpu)
            self.target_layer = ['conv_3', 'conv_7']

            for layer in self.target_layer:
                self.vgg.features[int(layer.split('_')[-1])].register_forward_hook(get_activation(layer))

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = osp.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = osp.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = osp.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = osp.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.test_step      = config['TRAINING_CONFIG']['TEST_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']

        self.lr_decay_step   = config['TRAINING_CONFIG']['LR_DECAY_STEP']
        self.lr_decay_policy = config['TRAINING_CONFIG']['LR_DECAY_POLICY']
        print(self.lr_decay_policy)

        if self.wandb:
            wandb.login(key='')
            wandb.init(project='StEP_training', name=self.train_dir)

        if self.use_tensorboard:
            self.build_tensorboard()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(osp.join(self.train_dir,'model_arch.txt'), 'w') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def get_state_dict(self, path):

        if path.startswith("module-"):
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(model_path, map_location=self.device)
            # https://github.com/computationalmedia/semstyle/issues/3
            # https://github.com/pytorch/pytorch/issues/10622
            # https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666/2
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            return new_state_dict
        else:
            return torch.load(path, map_location=lambda storage, loc: storage)

    def restore_model(self, epoch=None):

        if isinstance(epoch, int):
            ckpt_path = osp.join(self.model_dir, f'{epoch}-model.ckpt')
        else:
            ckpt_list = glob.glob(osp.join(self.model_dir, '*-model.ckpt'))

            if len(ckpt_list) == 0:
                return 0

            last_epoch = sorted([int(osp.basename(x).split('-')[0]) for x in ckpt_list])[-1]
            ckpt_path = osp.join(self.model_dir, f'{last_epoch}-model.ckpt')

        assert osp.exists(ckpt_path)

        print(f'ckpt path, {ckpt_path}, will be loaded ')
        ckpt = torch.load(ckpt_path)

        self.E.load_state_dict(ckpt['E'])
        self.D.load_state_dict(ckpt['D'])
        self.G.load_state_dict(ckpt['G'])

        if self.e_optimizer:
            self.e_optimizer.load_state_dict(ckpt['e_optim'])
        if self.g_optimizer:
            self.g_optimizer.load_state_dict(ckpt['g_optim'])
        if self.d_optimizer:
            self.d_optimizer.load_state_dict(ckpt['d_optim'])

        return epoch

    def build_stage1(self):

        if self.E_model == 'res18':
            self.E = models.resnet18(pretrained=True)
            in_features = self.E.fc.in_features
            self.E.fc = nn.Linear(in_features, self.num_style_feat)
        elif self.E_model == 'res50':
            self.E = models.resnet50(pretrained=True)
            in_features = self.E.fc.in_features
            self.E.fc = nn.Linear(in_features, self.num_style_feat)
        else:
            raise NotImplemented

        self.E = self.E.to(self.gpu)

        ckpt_list = sorted(glob.glob(osp.join(self.model_dir, 'stage1_*.ckpt')))
        epoch = None
        if len(ckpt_list):
            ckpt_path = ckpt_list[-1]

            epoch = int(osp.basename(ckpt_path).replace('.ckpt','').split('_')[-1])

            ckpt_dict = torch.load(ckpt_path, map_location=self.gpu)
            self.E.load_state_dict(ckpt_dict['E'])
            print(f'{ckpt_path} is load to E')

        self.e_optimizer = torch.optim.Adam(self.E.parameters(), self.e_lr, (self.beta1, self.beta2),
                                            weight_decay=self.lambda_e_regular)
        if self.lr_decay_policy == 'LambdaLR':
            self.e_scheduler = torch.optim.lr_scheduler.LambdaLR(self.e_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        elif self.lr_decay_policy == 'ExponentialLR':
            self.e_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.e_optimizer, gamma=0.5)

        if len(ckpt_list):
            return self.epoch1 - epoch
        else:
            return self.epoch1

    def build_stage2(self):

        self.E.eval()
        # build G and D
        self.G = UNet(n_channels=self.num_style_feat + 1).to(self.gpu)
        self.D = Discriminator().to(self.gpu)

        ckpt_list = sorted(glob.glob(osp.join(self.model_dir, 'stage2_*.ckpt')))
        epoch = None
        if len(ckpt_list):
            ckpt_path = ckpt_list[-1]
            epoch = int(osp.basename(ckpt_path).replace('.ckpt', '').split('_')[-1])
            ckpt_dict = torch.load(ckpt_path, map_location=self.gpu)
            self.G.load_state_dict(ckpt_dict['G'])
            self.D.load_state_dict(ckpt_dict['D'])
            print(f'{ckpt_path} is load to G and D')

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.e_lr, (self.beta1, self.beta2))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.e_lr, (0.0, self.beta2))

        if self.lr_decay_policy == 'LambdaLR':
            self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
            self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        elif self.lr_decay_policy == 'ExponentialLR':
            self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=0.5)
            self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=0.5)

        if len(ckpt_list):
            return self.epoch2 - epoch
        else:
            return self.epoch2

    def build_stage3(self):
        self.E.train()

        ckpt_list = sorted(glob.glob(osp.join(self.model_dir, 'stage3_*.ckpt')))
        epoch = None

        if len(ckpt_list):
            ckpt_path = ckpt_list[-1]
            epoch = int(osp.basename(ckpt_path).replace('.ckpt', '').split('_')[-1])
            ckpt_dict = torch.load(ckpt_path, map_location=self.gpu)
            self.G.load_state_dict(ckpt_dict['G'])
            self.D.load_state_dict(ckpt_dict['D'])
            self.E.load_state_dict(ckpt_dict['E'])
            print(f'{ckpt_path} is load to G and D')

        if len(ckpt_list):
            return self.epoch3 - epoch
        else:
            return self.epoch3

    def image_reporting(self, dir, fixed_edge, fixed_anchor, fixed_feat, epoch, postfix=''):
        image_report = list()
        image_report.append(fixed_edge.expand_as(fixed_anchor))
        image_report.append(fixed_anchor)
        image_report.append(self.G(torch.cat([fixed_feat, fixed_edge], dim=1)))
        x_concat = torch.cat(image_report, dim=3)
        epoch_str = str(epoch).zfill(len(str(self.epoch1 + self.epoch2 + self.epoch3)))
        sample_path = os.path.join(dir, '{}-images_{}.jpg'.format(epoch_str, postfix))
        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

    def train(self):
        # Set data loader.
        data_loader = self.train_data_loader
        iterations = len(self.train_data_loader)

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        f_anchor, f_close, f_far, f_edge = next(data_iter)
        f_anchor = f_anchor.to(self.gpu)
        f_close = f_close.to(self.gpu)
        f_far = f_far.to(self.gpu)
        f_edge = f_edge.to(self.gpu)

        splited_f_edge = list(torch.chunk(f_edge, self.batch_size, dim=0))
        first_fixed_edge = splited_f_edge[0]
        del splited_f_edge[0]
        splited_f_edge.append(first_fixed_edge)
        splited_f_edge = torch.cat(splited_f_edge, dim=0)

        start_time = time.time()
        print('Start training...')

        for epoch, stage in [[self.epoch1, 'stage1'], [self.epoch2, 'stage2'], [self.epoch3, 'stage3']]:

            if stage == 'stage1':
                epoch = self.build_stage1()
            elif stage == 'stage2':
                epoch = self.build_stage2()
            elif stage == 'stage3':
                self.build_stage3()

            if epoch == 0:
                print(f'{stage} is skipped')
                continue

            for e in range(epoch):

                for i in range(iterations):

                    loss = dict()

                    if stage in ['stage1', 'stage3']:
                        try:
                            anchor, close, far, _ = next(data_iter)
                        except:
                            data_iter = iter(data_loader)
                            anchor, close, far, _ = next(data_iter)

                        anchor = anchor.to(self.gpu)
                        close = close.to(self.gpu)
                        far = far.to(self.gpu)

                        feat_anchor = self.E(anchor)
                        feat_close = self.E(close)
                        feat_far = self.E(far)

                        triplet_loss = self.triplet_loss(feat_anchor, feat_close, feat_far)

                        loss['E/triplet_loss'] = triplet_loss.item()
                        triplet_loss.backward()
                        self.e_optimizer.step()

                    if stage in ['stage2', 'stage3']:
                        try:
                            anchor, _, _, edge = next(data_iter)
                        except:
                            data_iter = iter(data_loader)
                            anchor, _, _, edge = next(data_iter)

                        anchor = anchor.to(self.gpu)
                        edge = edge.to(self.gpu)

                        feat_anchor = self.E(anchor).unsqueeze(-1).unsqueeze(-1)
                        if stage == 'stage2':
                            feat_anchor = feat_anchor.detach()

                        batch, ed_ch, h, w = edge.size()
                        _, f_ch, _, _ = feat_anchor.size()
                        feat_anchor = feat_anchor.expand(batch, f_ch, h, w)

                        fake_feat = torch.cat([feat_anchor, edge], dim=1)
                        fake_images = self.G(fake_feat)

                        real_score = self.D(anchor)
                        fake_score = self.D(fake_images.detach())

                        d_loss_real = self.l2_loss(real_score, torch.ones_like(real_score))
                        d_loss_fake = self.l2_loss(fake_score, torch.ones_like(fake_score))

                        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake

                        if torch.isnan(d_loss):
                            raise Exception('{}, d_loss_fake is nan at Epoch [{}/{}] Iteration [{}/{}]'.format(stage, e + 1, epoch, i + 1, iterations))

                        self.d_optimizer.zero_grad()
                        d_loss.backward() # retain_graph=True
                        self.d_optimizer.step()

                        feat_anchor = self.E(anchor).unsqueeze(-1).unsqueeze(-1)
                        if stage == 'stage2':
                            feat_anchor = feat_anchor.detach()

                        batch, ed_ch, h, w = edge.size()
                        _, f_ch, _, _ = feat_anchor.size()
                        feat_anchor = feat_anchor.expand(batch, f_ch, h, w)

                        fake_feat = torch.cat([feat_anchor, edge], dim=1)
                        fake_images = self.G(fake_feat)
                        fake_score = self.D(fake_images)

                        g_loss_fake = self.l2_loss(fake_score, torch.ones_like(fake_score))
                        g_loss = self.lambda_g_fake * g_loss_fake

                        if self.percep:
                            fake_activation = dict()
                            real_activation = dict()

                            self.vgg(fake_images)
                            for layer in self.target_layer:
                                fake_activation[layer] = vgg_activation[layer]
                            vgg_activation.clear()

                            self.vgg(anchor)
                            for layer in self.target_layer:
                                real_activation[layer] = vgg_activation[layer]
                            vgg_activation.clear()

                            g_loss_percep = 0
                            for layer in self.target_layer:
                                g_loss_percep += self.l2_loss(fake_activation[layer], real_activation[layer])
                            g_loss += self.lambda_g_percep * g_loss_percep

                            fake_activation.clear()
                            real_activation.clear()

                        g_loss_recon = self.l1_loss(fake_images, anchor)
                        g_loss += self.lambda_g_recon * g_loss_recon

                        if torch.isnan(g_loss):
                            raise Exception('{}, d_loss_fake is nan at Epoch [{}/{}] Iteration [{}/{}]'.format(stage, e + 1, epoch, i + 1, iterations))

                        self.g_optimizer.zero_grad()
                        g_loss.backward()
                        self.g_optimizer.step()

                        loss['G/loss_fake'] = self.lambda_g_fake * g_loss_fake.item()
                        loss['G/loss_recon'] = self.lambda_g_recon * g_loss_recon.item()
                        if self.percep:
                            loss['G/loss_percep'] = self.lambda_g_percep * g_loss_percep.item()
                        loss['G/g_loss'] = g_loss.item()
                        loss['D/loss_real'] = self.lambda_d_real * d_loss_real.item()
                        loss['D/loss_fake'] = self.lambda_d_fake * d_loss_fake.item()
                        loss['D/d_loss'] = d_loss.item()

                    if self.wandb:
                        for tag, value in loss.items():
                            wandb.log({tag: value})

                    if (i + 1) % self.log_step == 0:
                        et = time.time() - start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]
                        log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, epoch, et, i + 1, iterations)
                        for tag, value in loss.items():
                            log += ", {}: {:.4f}".format(tag, value)

                        if self.use_tensorboard:
                            for tag, value in loss.items():
                                self.logger.scalar_summary(tag, value, e * epoch + i+1)

                        print(log)

                if (e + 1) % self.sample_step == 0:
                    with torch.no_grad():
                        if stage == 'stage1':
                            f_feat_anchor = self.E(f_anchor)
                            f_feat_close = self.E(f_close)
                            f_feat_far = self.E(f_far)
                            fixed_loss = self.lambda_e_triplet * self.triplet_loss(f_feat_anchor, f_feat_close, f_feat_far)
                            print(f'stage 1 fixed loss : {fixed_loss}')

                        elif stage in ['stage2', 'stage3']:
                            f_anchor_feat = self.E(f_anchor).unsqueeze(-1).unsqueeze(-1)
                            batch, ed_ch, h, w = f_edge.size()
                            _, f_ch, _, _ = f_anchor_feat.size()

                            if stage == 'stage2':
                                ep = self.epoch1 + e + 1
                            else:
                                ep = self.epoch1 + self.epoch2 + e + 1

                            f_anchor_feat = f_anchor_feat.expand(batch, f_ch, h, w)
                            self.image_reporting(self.sample_dir, f_edge, f_anchor, f_anchor_feat, ep)
                            self.image_reporting(self.sample_dir, splited_f_edge, f_anchor, f_anchor_feat, ep, 'shift')
                            print(f'Saved real and fake images into {self.sample_dir}...')

                # test step
                if (e + 1) % self.test_step == 0 and stage in ['stage2', 'stage3']:
                    self.test(stage, self.val_data_loader, e + 1, 'val')
                    self.test(stage, self.test_data_loader, e + 1, 'test')

                # Save model checkpoints.
                if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                    epoch_str = str(e + 1).zfill(len(str(self.epoch1 + self.epoch2 + self.epoch3)))
                    ckpt_path = osp.join(self.model_dir, f'{stage}_{epoch_str}.ckpt')
                    ckpt = dict()
                    if stage == 'stage1':
                        ckpt['E'] = self.E.state_dict()
                    elif stage == 'stage2':
                        ckpt['G'] = self.G.state_dict()
                        ckpt['D'] = self.D.state_dict()
                    elif stage == 'stage3':
                        ckpt['E'] = self.E.state_dict()
                        ckpt['G'] = self.G.state_dict()
                        ckpt['D'] = self.D.state_dict()
                    torch.save(ckpt, ckpt_path)
                    print('Saved model checkpoints into {}...'.format(self.model_dir))

                if (e + 1) % self.lr_decay_step == 0 and self.lr_decay_policy != 'None':
                    print('lr_scheduler is activated!!')
                    if self.e_scheduler and stage == 'stage1':
                        self.e_scheduler.step()
                    if self.g_scheduler and stage in ['stage2', 'stage3']:
                        self.g_scheduler.step()
                    if self.d_scheduler and stage in ['stage2', 'stage3']:
                        self.d_scheduler.step()
            print(f'{stage} is finished')

    def test(self, stage, data_loader, epoch, mode='train'):
        # Set data loader.

        target_dir = osp.join(self.result_dir, f'{mode}_{stage}_{epoch}')
        os.makedirs(target_dir, exist_ok=True)

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                anchor, _, _, edge = data
                anchor = anchor.to(self.gpu)
                edge = edge.to(self.gpu)

                splited_edge = list(torch.chunk(edge, self.batch_size, dim=0))
                first_fixed_edge = splited_edge[0]
                del splited_edge[0]
                splited_edge.append(first_fixed_edge)
                shifted_edge = torch.cat(splited_edge, dim=0)

                feat_anchor = self.E(anchor).unsqueeze(-1).unsqueeze(-1)
                batch, ed_ch, h, w = edge.size()
                _, f_ch, _, _ = feat_anchor.size()
                feat_anchor = feat_anchor.expand(batch, f_ch, h, w)
                self.image_reporting(target_dir, edge, anchor, feat_anchor, epoch, postfix=f'_{i}')
                self.image_reporting(target_dir, shifted_edge, anchor, feat_anchor, epoch, postfix=f'shift_{i}')
