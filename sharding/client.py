import os
import socket
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime
import logging

# logging 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 환경 변수 설정
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", 9090))
EPOCHS = int(os.getenv('EPOCHS', 10))
CLIENT_ID = int(os.getenv('CLIENT_ID', 1))
NUM_CLIENTS = int(os.getenv('NUM_CLIENTS', 5))

# CIFAR-10 데이터셋 로드
train_dataset_CIFAR10 = datasets.CIFAR10(
    root='data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

# 데이터 샤딩 함수


def get_sharded_dataset(dataset, num_clients, client_id):
    indices = list(range(len(dataset)))
    shard_size = len(dataset) // num_clients
    start_idx = (client_id - 1) * shard_size
    end_idx = start_idx + shard_size
    return Subset(dataset, indices[start_idx:end_idx])


# 클라이언트별로 샤드된 데이터셋 생성
train_dataset_sharded = get_sharded_dataset(
    train_dataset_CIFAR10, NUM_CLIENTS, CLIENT_ID)
train_loader_sharded = DataLoader(
    dataset=train_dataset_sharded,
    batch_size=64,
    shuffle=True
)

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


def resnet18(num_classes, grayscale):
    return ResNet_18(BasicBlock_18, [2, 2, 2, 2], num_classes, grayscale)


class Client:
    def __init__(self, server_address, max_retries=5, retry_interval=5):
        self.server_address = server_address
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_with_retries(max_retries, retry_interval)

    def connect_with_retries(self, max_retries, retry_interval):
        retries = 0
        while retries < max_retries:
            try:
                self.s.connect(self.server_address)
                logger.info("Connected to server.")
                break
            except ConnectionRefusedError:
                retries += 1
                logger.warning(
                    f"Connection refused, retrying... ({retries}/{max_retries})")
                time.sleep(retry_interval)
        if retries == max_retries:
            raise ConnectionError(
                "Max retries exceeded, could not connect to the server.")

    def log_message(self, message):
        logger.info(message)

    def get_id(self):
        data = self.s.recv(1024).decode('utf-8')
        client_id = int(data)
        logger.info(f"Client ID received: {client_id}")
        return client_id

    def recv_file(self, file_name):
        msg = self.s.recv(1024).decode('utf-8')
        self.s.sendall("ack\n".encode('utf-8'))
        if msg == "end":
            return msg
        with open(file_name, 'wb') as f:
            file_size = int.from_bytes(self.s.recv(8), byteorder='big')
            logger.info(f"Receiving file {file_name}, size: {file_size} bytes")
            received_size = 0
            while received_size < file_size:
                data = self.s.recv(4096)
                f.write(data)
                received_size += len(data)
        logger.info(f"File {file_name} received successfully.")
        return msg

    def send_file(self, file_name):
        self.s.sendall("READY_TO_SEND_FILE\n".encode('utf-8'))
        msg = self.s.recv(1024).decode('utf-8')
        if msg == "ack":
            with open(file_name, 'rb') as f:
                file_size = os.path.getsize(file_name)
                logger.info(
                    f"Sending file {file_name}, size: {file_size} bytes")
                self.s.sendall(file_size.to_bytes(8, byteorder='big'))
                data = f.read(4096)
                while data:
                    self.s.sendall(data)
                    data = f.read(4096)
            logger.info(f"File {file_name} sent successfully.")

    def learn(self, client_id, round):
        logger.info("Starting learning phase.")
        model = resnet18(10, False)
        model = torch.load('./global_model.pt')
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        for epoch in range(EPOCHS):
            total_loss, total_correct, total_samples = 0, 0, 0
            for i, (x, y) in enumerate(train_loader_sharded):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = nn.CrossEntropyLoss()(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                total_correct += (outputs.argmax(1) == y).sum().item()
                total_samples += x.size(0)
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            logger.info(
                f"Round {round}, Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        torch.save(model, f'client_model_{client_id}.pt')
        logger.info("Learning phase completed.")

    def get_updated_model(self, client_id, round):
        msg = self.recv_file("global_model.pt")
        if msg == "end":
            return "end"
        self.learn(client_id, round)
        self.send_file(f"client_model_{client_id}.pt")
        self.s.sendall("done learning\n".encode('utf-8'))

    def run(self):
        logger.info("Client is starting.")
        client_id = self.get_id()
        round = 1
        while True:
            logger.info(f"Round {round} started.")
            status = self.get_updated_model(client_id, round)
            round += 1
            if status == "end":
                logger.info("Received termination signal. Exiting.")
                self.s.sendall("end".encode('utf-8'))
                break


if __name__ == "__main__":
    client = Client(server_address=(SERVER_ADDRESS, SERVER_PORT))
    client.run()
