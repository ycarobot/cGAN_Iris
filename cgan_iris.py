import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import json

# from matlab_dataset_cgan import MatLabDataset
from json_dataset_cgan import JSONDataset

from sklearn.datasets import load_iris
data = load_iris()
import pandas as pd
#直接读到pandas的数据框中
pd.DataFrame(data=data.data, columns=data.feature_names)
import matplotlib.pyplot as plt
plt.style.use('ggplot')


X = data.data  # 只包括样本的特征，150x4
y = data.target  # 样本的类型，[0, 1, 2]
# print(len(X)): 150
# print(len(y)): 150
# features = data.feature_names  # 4个特征的名称
# targets = data.target_names  # 3类鸢尾花的名称，跟y中的3个数字对应

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=150, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=4, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=3, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)     #opt.img_size=32

cuda = True if torch.cuda.is_available() else False

print("np.prod(img_shape) is: %d" %int(np.prod(img_shape)) )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 16, normalize=False),      #128
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
            nn.Linear(128, 4),     # int(np.prod(img_shape))=1024
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        print("self.label_emb(labels):")
        print(self.label_emb(labels).size())
        print("noise: ")
        print(noise.size())
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        print("gen_input: ")
        print(gen_input.size())
        img = self.model(gen_input)
        print("img: ")
        print(img.size())
        img = img.view(img.size(0), *img_shape)
        print("size of img: ")
        print(img.size())
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + 4, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 16),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 16),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        print("d_in: ")
        print(d_in.size())
        validity = self.model(d_in)
        print("validity: ")
        print(validity.size())
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    JSONDataset("iris_Dataset_data.json","iris_Dataset_label.json"),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    print("z: ")
    print(z.size())
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    print("labels: ")
    # print(labels.size())
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    print("gen_imgs: ")
    print(gen_imgs.size())
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------
result=[]
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        print("The size of  (epoch) : " )
        print(imgs.size())
        print("The size of labels (epoch) : ")
        print(labels.size())
        batch_size = imgs.shape[0] #64

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        print("The size of valid: ")
        print(valid.size())
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        print("The size of fake: ")
        print(fake.size())

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        print("The size of real_imgs: ")
        print(real_imgs.size())
        labels = Variable(labels.type(LongTensor))
        print("The size of labels (epoch . Configure input) :　")
        print(labels.size())

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        print("The size of z (epoch): ")
        print(z.size())
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        print("The size of gen_labels (epoch): ")
        print(gen_labels.size())

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        print("The size of gen_imgs (epoch . Generate a batch of images ):")
        print(gen_imgs.size())
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        print("The size of validity (epoch) : ")
        print(validity.size())
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        print("The size of validity_real (epoch . Loss for real images) :　")
        print(validity_real.size())
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        if batches_done % opt.sample_interval == 0 and batches_done>0:
            result.append(gen_imgs[0].cpu())
            result_tensor = torch.stack(result)
            # sample_image(n_row=10, batches_done=batches_done)
result_numpy=result_tensor.detach().numpy()
io.savemat('Data_Generated_Source_cGAN.mat', {'result_numpy': result_numpy})
