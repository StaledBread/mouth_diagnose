import torch
import torch.nn as nn
import torch.optim as optim



#combined network
class comn_01(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(comn_01, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5, stride=3, padding=0),
            nn.ReLU(True),
            nn.Conv2d(16, 4, 7, stride=3, padding=3),
            nn.ReLU(True),
            nn.Conv2d(4, 1, 3, stride=1, padding=0),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(5, 10),
            nn.PReLU(),
            nn.Linear(10, 10),
            nn.PReLU(),
            nn.Linear(10, 9),
            nn.Sigmoid()
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(18, 18),
            nn.PReLU(),
            nn.Linear(18, 18),
            nn.PReLU(),
            nn.Linear(18, out_dim),
            nn.Sigmoid()
        )

    def forward(self, img, spt):
        img_out = self.conv(img)
        img_out = img_out.view(-1)
        spt_out_1 = self.fc_1(spt)
        hidden_input = torch.cat((img_out, spt_out_1))
        out = self.fc_2(hidden_input)
        return out

model = comn_01(3, 10)  # The size of the image should be 50×50
use_gpu = torch.cuda.is_available()  # check GPU
if use_gpu:
    model = model.cuda()
# initialize criterion and optimizer
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)



# a = torch.rand(3, 4, 5)
# b = a.view(-1)
# c = torch.rand(60)
# d = torch.cat((c, b))
# print(a.size(), b.size(), c.size(), d.size())
