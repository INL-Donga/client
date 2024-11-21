import socket
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
import time
# from torch.utils.tensorboard import SummaryWriter


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS")
SERVER_PORT = int(os.getenv("SERVER_PORT"))
EPOCHS = int(os.getenv('EPOCHS'))

if EPOCHS is None:
    EPOCHS = 10

# 데이터셋 로드
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


# ResNet18 모델 정의
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


class ComplexCNN(nn.Module):
    def __init__(self, grayscale):
        if grayscale:
            in_dim = 1
            self.out_size = 7
        else:
            in_dim = 3
            self.out_size = 8
        super(ComplexCNN, self).__init__()

        self.conv1_r = nn.Conv2d(
            in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv1_i = nn.Conv2d(
            in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)

        self.conv2_r = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2_i = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_r = nn.Linear(self.out_size*self.out_size*32, 32)
        self.fc1_i = nn.Linear(self.out_size*self.out_size*32, 32)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x_real = self.conv1_r(x) - self.conv1_i(x)
        x_imag = self.conv1_r(x) + self.conv1_i(x)

        x_real = torch.relu(x_real)
        x_imag = torch.relu(x_imag)

        x_real = self.pool(x_real)
        x_imag = self.pool(x_imag)

        x_real_2 = self.conv2_r(x_real) - self.conv2_i(x_imag)
        x_imag_2 = self.conv2_r(x_imag) + self.conv2_i(x_real)

        x_real_2 = torch.relu(x_real_2)
        x_imag_2 = torch.relu(x_imag_2)

        x_real_2 = self.pool(x_real_2)
        x_imag_2 = self.pool(x_imag_2)

        x_real_2 = x_real_2.view(-1, self.out_size*self.out_size * 32)
        x_imag_2 = x_imag_2.view(-1, self.out_size*self.out_size * 32)

        x_real_fc1 = self.fc1_r(x_real_2) - self.fc1_i(x_imag_2)
        x_imag_fc1 = self.fc1_r(x_imag_2) + self.fc1_i(x_real_2)

        x_real_fc1 = torch.relu(x_real_fc1)
        x_imag_fc1 = torch.relu(x_imag_fc1)

        combined = torch.cat([x_real_fc1, x_imag_fc1], dim=1)

        output = self.fc2(combined)

        return output


class Client:
    def __init__(self, server_address=(SERVER_ADDRESS, SERVER_PORT), max_retries=5, retry_interval=5):
        self.server_address = server_address
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_with_retries(max_retries, retry_interval)

    def connect_with_retries(self, max_retries, retry_interval):
        retries = 0
        while retries < max_retries:
            try:
                self.s.connect(self.server_address)
                print("Connected to server.")
                break
            except ConnectionRefusedError:
                retries += 1
                print(
                    f"Connection refused, retrying... ({retries}/{max_retries})")
                time.sleep(retry_interval)
        if retries == max_retries:
            raise ConnectionError(
                "Max retries exceeded, could not connect to the server.")

    def log_message(self, message):
        current_time = datetime.now().strftime("%M:%S.%f")[:-3]
        print(f"[{current_time}] {message}")

    def get_id(self):
        data = self.s.recv(1024).decode('utf-8')
        client_id = int(data)
        self.log_message(f"client id : {client_id}")
        return client_id

    def recv_file(self, file_name):
        msg = self.s.recv(1024).decode('utf-8')
        self.s.sendall("ack\n".encode('utf-8'))
        if msg == "end":
            return msg
        with open(file_name, 'wb') as f:
            file_size = int.from_bytes(self.s.recv(8), byteorder='big')
            self.log_message(f"file size : {file_size}")
            received_size = 0
            while received_size < file_size:
                data = self.s.recv(4096)
                f.write(data)
                received_size += len(data)
        self.log_message(f"{file_name} received and saved")
        return msg

    def send_file(self, file_name):
        self.s.sendall("READY_TO_SEND_FILE\n".encode('utf-8'))
        msg = self.s.recv(1024).decode('utf-8')
        if msg == "ack":
            with open(file_name, 'rb') as f:
                file_size = os.path.getsize(file_name)
                self.log_message(
                    f"Sending {file_name} to server, size : {file_size}bytes")
                self.s.sendall(file_size.to_bytes(8, byteorder='big'))
                data = f.read(4096)
                while data:
                    self.s.sendall(data)
                    data = f.read(4096)
            self.log_message(f"{file_name} sent to server")

    def get_layer_parameters(self, model, round):
        if round % 2 == 0:
            train_weight = {name: param for name, param in model.state_dict().items(
            ) if 'conv1_i' in name or 'conv2_i' in name or 'fc1_i' in name or 'fc2' in name}
        else:
            train_weight = {name: param for name, param in model.state_dict().items(
            ) if 'conv1_r' in name or 'conv2_r' in name or 'fc1_r' in name or 'fc2' in name}

        return train_weight

    def learn(self, client_id, round):
        self.log_message("Start Learning")
        local_model = ComplexCNN(False)
        local_model = torch.load('./global_model.pt')
        local_model.to(device)
        local_model.train()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)

        if round % 2 == 0:
            for param in local_model.conv1_r.parameters():
                param.requires_grad = False

            for param in local_model.conv2_r.parameters():
                param.requires_grad = False

            for param in local_model.fc1_r.parameters():
                param.requires_grad = False
        else:
            for param in local_model.conv1_i.parameters():
                param.requires_grad = False

            for param in local_model.conv2_i.parameters():
                param.requires_grad = False

            for param in local_model.fc1_i.parameters():
                param.requires_grad = False

        # 학습 과정 모니터링을 위한 변수 초기화
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(EPOCHS):
            for i, (x, y) in enumerate(train_loader_CIFAR10):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                local_out = local_model(x)
                loss = nn.CrossEntropyLoss()(local_out, y)
                loss.backward()
                optimizer.step()

                # 미니배치별 손실 출력
                # print(
                #     f"[Round {round} - Epoch {epoch + 1} - Batch {i + 1}] Loss: {loss.item():.4f}")

                total_loss += loss.item() * x.size(0)
                total_correct += (local_out.argmax(1) == y).sum().item()
                total_samples += x.size(0)

            # 에포크 종료 시 평균 손실 및 정확도 계산 및 출력
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print(
                f"[Round {round} - Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            self.log_message(
                f'Round {round}  Client {client_id} Training Success - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        model_parameters = self.get_layer_parameters(local_model, round)
        torch.save(model_parameters, f'client_model_{client_id}.pt')
        self.log_message("End Learning")

    def get_updated_model(self, client_id, round):
        msg = self.recv_file("global_model.pt")
        if msg == "end":
            return "end"
        self.learn(client_id, round)
        self.send_file(f"client_model_{client_id}.pt")
        self.s.sendall("done learning\n".encode('utf-8'))

    def run(self):
        self.log_message("시작")
        self.log_message(f"서버에 접속 중: {self.s.getpeername()}")
        client_id = self.get_id()
        round = 1
        while True:
            self.log_message(f"Round {round} start")
            status = self.get_updated_model(client_id, round)
            round += 1
            if status == "end":
                self.log_message("종료 코드 수신")
                self.s.sendall("end".encode('utf-8'))
                break


if __name__ == "__main__":
    client = Client()
    client.run()
