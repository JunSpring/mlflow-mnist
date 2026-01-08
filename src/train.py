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
import os
import dvc.api
import yaml
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST with MLflow")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--train_size", type=int, default=10000, help="Subset of training data")
    parser.add_argument("--tracking_uri", type=str, default=None)
    return parser.parse_args()

def verify_environment_clean():
    """Git과 DVC가 클린한 상태인지 검사합니다."""
    print("🔍 환경 상태 검사 중...")

    # 1. Git 상태 체크 (수정된 코드나 커밋되지 않은 data.dvc 확인)
    try:
        git_status = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        if git_status:
            print("\n❌ [ERROR] Git 상태가 Dirty합니다! 변경사항을 커밋하세요.")
            print(f"--- 수정된 파일 목록 ---\n{git_status}\n")
            return False
    except Exception as e:
        print(f"⚠️ Git 상태를 확인할 수 없습니다: {e}")
        return False

    # 2. DVC 상태 체크 (실제 데이터 실물이 .dvc 파일의 해시와 일치하는지 확인)
    try:
        # dvc status가 아무것도 출력하지 않으면 클린한 상태입니다.
        dvc_status = subprocess.check_output(["dvc", "status", "--quiet"])
        # dvc status는 변경사항이 있으면 에러 코드(non-zero)를 반환하거나 메시지를 출력합니다.
    except subprocess.CalledProcessError:
        print("\n❌ [ERROR] DVC 데이터 상태가 Dirty합니다! 'dvc commit' 또는 'dvc add'를 수행하세요.")
        return False
    except Exception as e:
        print(f"⚠️ DVC 상태를 확인할 수 없습니다: {e}")
        return False

    print("✅ 환경이 깨끗합니다. 학습을 시작합니다.")
    return True

def get_dvc_hash(dvc_file_path='data.dvc'):
    """로컬의 .dvc 파일을 직접 읽어 MD5 해시값을 추출합니다."""
    # mlflow run 실행 시 파일 경로를 찾기 위해 절대 경로로 변환
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, dvc_file_path)
    
    try:
        with open(full_path, 'r') as f:
            dvc_data = yaml.safe_load(f)
            # .dvc 파일의 outs 리스트에서 md5 값을 가져옴
            return dvc_data['outs'][0]['md5']
    except Exception as e:
        print(f"DVC 메타데이터 읽기 실패: {e}")
        return "unknown"

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

def setup_mlflow(tracking_uri, experiment_name, run_name=None):
    mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)
    
    mlflow.enable_system_metrics_logging()
    mlflow.autolog()

    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        
        print(f"MLflow: logging run_id({active_run.info.run_id}) to {tracking_uri}")
        return active_run
        
    except Exception as e:
        print(f"MLflow: Failed to initialize: {e}")
        return None

def main():
    if not verify_environment_clean():
        print("🛑 재현성을 위해 더티 상태에서는 실행할 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1) # 에러 코드를 남기고 강제 종료
        
    args = parse_args()

    dataset_version = get_dvc_hash()

    data_path = '/home/junspring/mlflow-mnist/data'
    data_path_uri = f"file://{os.path.abspath(data_path)}"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
    train_dataset = Subset(full_train_dataset, np.arange(args.train_size))
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=False, transform=transform)
    test_dataset = Subset(test_dataset, np.arange(1000))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    train_ds = mlflow.data.from_numpy(
            features=full_train_dataset.data[:args.train_size].numpy(),
            targets=full_train_dataset.targets[:args.train_size].numpy(),
            name="mnist_train_subset",
            source=f"file://{os.path.abspath(data_path)}",
            digest=dataset_version  # MLflow 데이터셋 다이제스트로 사용
        )

    myNeuralNet = NeuralNet()
    myOptimizer = torch.optim.Adam(myNeuralNet.parameters(), lr=args.lr)

    tracking_uri = args.tracking_uri or mlflow.get_tracking_uri()
    run = setup_mlflow(tracking_uri, "MLflow MNIST Test")

    if run:
        with run:
            mlflow.log_input(train_ds, context="training")
            # 모든 매개변수 자동 기록
            mlflow.log_params(vars(args))
            mlflow.set_tag("dvc.dataset_version", dataset_version)
            
            for epoch in range(args.epochs):
                train(myNeuralNet, train_loader, myOptimizer, epoch, log_interval=40)

            # 모델 Signature 및 샘플 데이터 설정
            input_example = next(iter(train_loader))[0][:1].numpy()
            signature = infer_signature(input_example, myNeuralNet(torch.tensor(input_example)).detach().numpy())

            # 모델 저장 (MLflow 가이드 방식)
            mlflow.pytorch.log_model(
                pytorch_model=myNeuralNet,
                name="model",
                signature=signature,
                input_example=input_example
            )
            print(f"Run ID: {run.info.run_id} 완료!")

if __name__ == "__main__":
    main()