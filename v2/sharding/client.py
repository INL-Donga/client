import socket
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
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
EPOCHS = int(os.getenv("EPOCHS", 10))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", 3))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.01))
MOMENTUM = float(os.getenv("MOMENTUM", 0.9))


# CIFAR-10 데이터셋 로드 및 샤딩
train_dataset_CIFAR10 = datasets.CIFAR10(
    root='data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)


def get_sharded_dataset(dataset, num_clients, client_id):
    indices = list(range(len(dataset)))
    shard_size = len(dataset) // num_clients
    start_idx = (client_id - 1) * shard_size
    end_idx = start_idx + shard_size
    logger.info(f"Client {client_id}: Dataset range [{start_idx}, {end_idx})")
    return Subset(dataset, indices[start_idx:end_idx])


# ComplexCNN 정의
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
        self.client_id = None  # 클라이언트 ID 초기화
        self.train_loader_sharded = None  # 샤딩된 데이터 로더 초기화

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

    def get_id(self):
        data = self.s.recv(1024).decode('utf-8')
        client_id = int(data)
        logger.info(f"Client ID received: {client_id}")
        self.client_id = client_id

        # 클라이언트 ID로 데이터셋 샤딩 설정
        train_dataset_sharded = get_sharded_dataset(
            train_dataset_CIFAR10, NUM_CLIENTS, self.client_id)
        self.train_loader_sharded = DataLoader(
            dataset=train_dataset_sharded,
            batch_size=64,
            shuffle=True
        )

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

    def get_layer_parameters(self, model, round):
        if round % 2 == 0:
            train_weight = {name: param for name, param in model.state_dict().items(
            ) if 'conv1_i' in name or 'conv2_i' in name or 'fc1_i' in name or 'fc2' in name}
        else:
            train_weight = {name: param for name, param in model.state_dict().items(
            ) if 'conv1_r' in name or 'conv2_r' in name or 'fc1_r' in name or 'fc2' in name}

        return train_weight

    def learn(self, round):
        logger.info("Starting learning phase.")
        local_model = ComplexCNN(False)
        local_model.load_state_dict(torch.load('./global_model.pt'))
        local_model.to(device)
        local_model.train()
        optimizer = torch.optim.SGD(
            local_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

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

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(EPOCHS):
            for i, (x, y) in enumerate(self.train_loader_sharded):
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
            logger.info(
                f'Round {round} - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        model_parameters = self.get_layer_parameters(local_model, round)
        logging.info(f'test: {self.client_id}')
        torch.save(model_parameters, f'client_model_{self.client_id}.pt')
        logger.info("Learning phase completed.")

    def get_updated_model(self, round):
        msg = self.recv_file("global_model.pt")
        if msg == "end":
            return "end"
        self.learn(round)
        self.send_file(f'client_model_{self.client_id}.pt')
        self.s.sendall("done learning\n".encode('utf-8'))

    def run(self):
        logger.info("Client is starting.")
        self.get_id()
        round = 1
        while True:
            logger.info(f"Starting round {round}")
            status = self.get_updated_model(round)
            round += 1
            if status == "end":
                logger.info("Termination signal received. Exiting.")
                self.s.sendall("end".encode('utf-8'))
                break


if __name__ == "__main__":
    client = Client()
    client.run()
