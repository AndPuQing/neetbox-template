import torchvision
from torch.utils.data import DataLoader
from omegaconf import DictConfig


def get_train_dataloader(cfg: DictConfig):
    train_dataset = torchvision.datasets.MNIST(
        root=cfg.data.root,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    return train_loader


def get_test_dataloader(cfg):
    test_dataset = torchvision.datasets.MNIST(
        root=cfg.data.root,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    return test_loader


def get_dataloader(cfg):
    train_loader = get_train_dataloader(cfg)
    test_loader = get_test_dataloader(cfg)
    return train_loader, test_loader
