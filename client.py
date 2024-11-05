import socket
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_dataset_CIFAR10 = datasets.CIFAR10(root='data',
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         download=True)
test_dataset_CIFAR10 = datasets.CIFAR10(root='data',
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)
train_loader_CIFAR10 = DataLoader(dataset=train_dataset_CIFAR10,
                                  batch_size=64,
                                  shuffle=True)
test_loader_CIFAR10 = DataLoader(dataset=test_dataset_CIFAR10,
                                 batch_size=64,
                                 shuffle=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock_18(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_18, self).__init__()
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(planes, planes)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet_18(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        in_dim = 1 if grayscale else 3
        super(ResNet_18, self).__init__()
        self.conv1g = nn.Conv2d(in_dim, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bng = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend([block(self.inplanes, planes) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1g(x)
        x = self.bng(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_full = x.view(x.size(0), -1)
        logits = self.fc(x_full)
        return logits, x_full


def resnet18(num_classes, grayscale):
    return ResNet_18(block=BasicBlock_18, layers=[2, 2, 2, 2], num_classes=num_classes, grayscale=grayscale)


criterion = nn.CrossEntropyLoss()


def avg_train_client(id, client_loader, global_model, num_local_epochs, lr):
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(num_local_epochs):
        print(f'    Epoch {epoch + 1}')
        for (i, (x, y)) in enumerate(client_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            local_out, _ = local_model(x)
            loss = criterion(local_out, y)
            loss.backward()
            optimizer.step()
    return local_model


def log_message(message):
    current_time = datetime.now().strftime("%M:%S.%f")[:-3]
    print(f"[{current_time}] {message}")


server_address = (os.getenv('SERVER_ADDRESS', '127.0.0.1'),
                  int(os.getenv('SERVER_PORT', '9090')))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(server_address)


def getId():
    data = s.recv(1024).decode('utf-8')
    id = int(data)
    log_message(f"client id : {id}")
    return id


def recvFile(fileName):
    msg = s.recv(1024).decode('utf-8')
    s.sendall("ack\n".encode('utf-8'))
    if msg == "end":
        return msg
    with open(fileName, 'wb') as f:
        file_size = int.from_bytes(s.recv(8), byteorder='big')
        log_message(f"file size : {file_size}")
        received_size = 0
        while received_size < file_size:
            data = s.recv(4096)
            f.write(data)
            received_size += len(data)
    log_message(f"{fileName} received and saved")
    return msg


def sendFile(fileName):
    s.sendall("READY_TO_SEND_FILE\n".encode('utf-8'))
    msg = s.recv(1024).decode('utf-8')
    if msg == "ack":
        with open(fileName, 'rb') as f:
            file_size = os.path.getsize(fileName)
            log_message(
                f"Sending {fileName} to server, size : {file_size}bytes")
            s.sendall(file_size.to_bytes(8, byteorder='big'))
            data = f.read(4096)
            while data:
                s.sendall(data)
                data = f.read(4096)
        log_message(f"{fileName} sent to server")


def Learning(id, round):
    log_message("Start Learning")
    model = resnet18(10, False)
    model = torch.load('./global_model.pt')
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(1):
        for (i, (x, y)) in enumerate(train_loader_CIFAR10):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            local_out, _ = model(x)
            loss = criterion(local_out, y)
            loss.backward()
            optimizer.step()
        log_message(f'Round {round}  Client {id} Training Success')
    torch.save(model, f'client_model_{id}.pt')
    log_message("End Learning")


def getUpdatedPT(id, round):
    msg = recvFile("global_model.pt")
    if msg == "end":
        return "end"
    Learning(id, round)
    sendFile(f"client_model_{id}.pt")
    s.sendall("done learning\n".encode('utf-8'))


def run():
    log_message("시작")
    log_message(f"서버에 접속 중: {s.getpeername()}")
    id = getId()
    round = 1
    while True:
        log_message(f"Round {round} start")
        t = getUpdatedPT(id, round)
        round += 1
        if t == "end":
            log_message("종료 코드 수신")
            s.sendall("end".encode('utf-8'))
            break


if __name__ == "__main__":
    run()
