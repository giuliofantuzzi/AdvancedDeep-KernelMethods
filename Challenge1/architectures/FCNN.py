import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim, non_linearity,dropout):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights with N(0, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.non_linearity(self.fc1(x))
        x = self.dropout(x)
        x = self.non_linearity(self.fc2(x))
        x = self.fc3(x)
        return x