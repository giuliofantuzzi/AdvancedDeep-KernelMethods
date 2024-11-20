import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim,non_linearity,pooling,batchnorm):
        super(CNN, self).__init__()
        self.input_dim = input_dim   #28
        self.batchnorm = batchnorm
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.pooling = pooling
        self.non_linearity = non_linearity
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        
        # Initialize weights with N(0, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=1)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # Reshape input to a suitable format for the convolutions
        x = x.view(-1, 1, self.input_dim, self.input_dim)             # [batch_size, 1, 28, 28] 
        
        # 1st Convolutional Layer
        x = self.conv1(x)                                             # [batch_size, 16, 28, 28]  
        if self.batchnorm:
            x = self.batchnorm1(x)                                    # [batch_size, 16, 28, 28]                  
        x = self.non_linearity(x)                                     # [batch_size, 16, 28, 28]               
        x = self.pooling(x)                                           # [batch_size, 16, 14, 14]
        
        # 2nd Convolutional Layer
        x = self.conv2(x)                                             # [batch_size, 32, 14, 14]               
        if self.batchnorm:
            x = self.batchnorm2(x)                                    # [batch_size, 32, 14, 14]
        x = self.non_linearity(x)                                     # [batch_size, 32, 14, 14]
        x = self.pooling(x)                                           # [batch_size, 32, 7, 7]
        
        # Reshape to a suitable format for the fully connected layers
        x = x.view(-1, 32*7*7)                                        # [batch_size, 32*7*7]
        x = self.fc1(x)                                               # [batch_size, 128]
        x = self.non_linearity(x)                                     # [batch_size, 128]
        x = self.fc2(x)                                               # [batch_size, output_dim]
        
        return x