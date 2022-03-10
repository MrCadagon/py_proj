# google net
# residual net
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist', train=True, download=False, transform=transforms)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, download=False, transform=transforms)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

        self.branch1X1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5X5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5X2_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3X3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3X3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3X3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branch1X1 = self.branch1X1(x)

        branch5X5 = self.branch5X2_2(self.branch5X5_1(x))

        branch3X3 = self.branch3X3_3(self.branch3X3_2(self.branch3X3_1(x)))

        return torch.cat([branch_pool, branch1X1, branch3X3, branch5X5], dim=1)  #


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        # 24*24->12*12
        self.mp = nn.MaxPool2d(2)

        self.inception_A = Inception(10)
        self.inception_B = Inception(20)

        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.inception_A(x)

        x = F.relu(self.mp(self.conv2(x)))
        x = self.inception_B(x)

        x = x.view(in_size, -1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        return F.relu(self.conv1(y) + x)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.mp=nn.MaxPool2d(2)

        self.residual_A=ResidualBlock(16)
        self.residual_B=ResidualBlock(32)

        self.fc=nn.Linear(512,10)

    def forward(self, x):
        in_size = x.size(0)

        x=self.conv1(x)
        x=F.relu(x)
        x=self.mp(x)
        x=self.residual_A(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.residual_B(x)

        x = x.view(in_size, -1)

        x = self.fc(x)

        return x


def train(epoch, model_now,optimizer):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model_now(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test(model_now):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model_now(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':

    model = Net1()
    model_2 = Net2()
    # construct loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    optimizer_2 = optim.SGD(model_2.parameters(), lr=0.01, momentum=0.5)
    # for epoch in range(10):
    #     train(epoch, model)
    #     test(model)
    for epoch in range(10):
        train(epoch, model_2,optimizer_2)
        test(model_2)