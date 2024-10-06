import torch.nn as nn

class FCDiscriminator(nn.Module):
    """
    inplanes, planes. Patch-gan
    """

    def __init__(self, inplanes, planes = 64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(planes*4, planes*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(planes*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x