import hydra
from omegaconf import DictConfig
from model import Net
from torch import nn
from torch import optim
import mlflow
from dataset import get_dataloader
from utils import log_params_from_omegaconf_dict
import torch


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    model = Net(cfg)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum
    )

    train_loader, test_loader = get_dataloader(cfg)

    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(cfg.mlflow.runname)
    with mlflow.start_run():
        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            running_loss = 0.0
            log_params_from_omegaconf_dict(cfg)
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_value = loss(outputs, labels)
                loss_value.backward()
                optimizer.step()

                running_loss += loss_value.item()
                mlflow.log_metric("train_loss", running_loss)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = float(correct / total)
    return accuracy


if __name__ == "__main__":
    train()
