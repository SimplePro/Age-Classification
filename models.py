import torch.nn as nn
from torchsummary import summary



class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, last_block=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.std = 2 / (in_channels * kernel_size**2)
        nn.init.normal_(self.conv.weight, mean=0.0, std=self.std)
        nn.init.zeros_(self.conv.bias)
        

        self.layer = nn.ModuleList([
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        ])
        
        if not last_block:
            self.layer.append(nn.MaxPool2d(kernel_size=2, stride=2))


    def forward(self, x):
        
        out = self.conv(x)

        for i in range(len(self.layer)):
            out = self.layer[i](out)

        return out


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, init="he"):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        if init == "he":
            self.std = 2 / (in_features)

        elif init == "glorot":
            self.std = 1 / ((in_features + out_features) / 2)

        nn.init.normal_(self.linear.weight, mean=0.0, std=self.std)
        nn.init.zeros_(self.linear.bias)

    
    def forward(self, x):
        out = self.linear(x)

        return out



class AgeClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        channels_pair = [(3, 16), (16, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]

        self.convolutional_layer = nn.ModuleList()

        for pair in channels_pair:

            if pair[0] == 512:
                self.convolutional_layer.append(ConvBlock(pair[0], pair[1], kernel_size=4, stride=1, padding=0, last_block=True))

            else:
                self.convolutional_layer.append(ConvBlock(pair[0], pair[1]))

        
        self.fc = nn.Sequential(
            nn.Flatten(),

            LinearLayer(1024, 100, init="he"),
            nn.LeakyReLU(0.2),

            LinearLayer(100, 1, init="glorot"),
            nn.Sigmoid()
        )


    def forward(self, x):
        out = x

        for i in range(len(self.convolutional_layer)):
            out = self.convolutional_layer[i](out)
        
        out = self.fc(out)

        return out



if __name__ == '__main__':
    age_classifier = AgeClassifier().cuda()
    summary(age_classifier, (3, 128, 128))