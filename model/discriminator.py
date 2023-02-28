import torch
import torch.nn as nn
import torch.nn.functional as F
from rsaModules_v2 import rsaBlock


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		#self.rsaBlock = fmap_FCDiscriminator(in_channal=337)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
######################################################
		#self.conv1 = nn.Conv2d(337, 128, kernel_size=4, stride=2, padding=1)
		#self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1)

		#self.conv3 = nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1)
		#self.conv4 = nn.Conv2d(64, 16, kernel_size=4, stride=2, padding=1)

		#self.classifier = nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		#print(mask.shape)
		#print(fmap.shape)
		#x = self.rsaBlock(x)
		#fmap = F.interpolate(fmap, (400, 400), mode='bilinear')
		#x = torch.cat([mask, fmap], 1)
		#print(x.shape)

		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		#x = self.rsa_block(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x


class fmap_FCDiscriminator(nn.Module):

	def __init__(self, in_channal):
		super(fmap_FCDiscriminator, self).__init__()

		self.rsaBlock = rsaBlock(in_channal)


	def forward(self, x):

		x = self.rsaBlock(x)

		return x

