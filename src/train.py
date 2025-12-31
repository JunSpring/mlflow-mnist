import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import argparse
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from torch.utils.data import Subset

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST with MLflow")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--train_size", type=int, default=10000, help="Subset of training data")
    parser.add_argument("--tracking_uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    return parser.parse_args()

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        # (1, 28, 28) -> (16, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        
        # (16, 28, 28) -> (32, 28, 28)
        x = self.conv2(x)
        x = F.relu(x)
        
        # (32, 28, 28) -> (32, 14, 14)
        x = F.max_pool2d(x, 2)

        # (32, 14, 14) -> (6272, 1)
        x = torch.flatten(x, 1)

        # (6272, 1) -> (128, 1)
        x = self.fc1(x)
        x = F.relu(x)
        
        # (128, 1) -> (10, 1)
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        return output
            
def train(model, train_loader, optimizer, epoch, log_interval):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)    
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            # 배치별 Loss 기록
            step = epoch * len(train_loader) + batch_idx
            mlflow.log_metric("batch_loss", loss.item(), step=step)

    avg_loss = total_loss / len(train_loader)
    mlflow.log_metric("avg_train_loss", avg_loss, step=epoch)

def main():
    args = parse_args()
    
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("MLflow MNIST Test")
    mlflow.enable_system_metrics_logging()

    data_path = '/home/junspring/mlflow-mnist/data'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
    train_dataset = Subset(full_train_dataset, np.arange(args.train_size))
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=False, transform=transform)
    test_dataset = Subset(test_dataset, np.arange(1000))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    myNeuralNet = NeuralNet()
    myOptimizer = torch.optim.Adam(myNeuralNet.parameters(), lr=args.lr)

    # MLflow 실행 시작
    with mlflow.start_run() as run:
        # 모든 매개변수 자동 기록
        mlflow.log_params(vars(args))
        
        for epoch in range(args.epochs):
            train(myNeuralNet, train_loader, myOptimizer, epoch, log_interval=40)

        # 모델 Signature 및 샘플 데이터 설정
        input_example = next(iter(train_loader))[0][:1].numpy()
        signature = infer_signature(input_example, myNeuralNet(torch.tensor(input_example)).detach().numpy())

        # 모델 저장 (MLflow 가이드 방식)
        mlflow.pytorch.log_model(
            pytorch_model=myNeuralNet,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        print(f"Run ID: {run.info.run_id} 완료!")

if __name__ == "__main__":
    main()