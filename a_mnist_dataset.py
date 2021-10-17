# %%
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# %%
'''MNIST DataModule'''


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, dataset_dir, train_batch_size, test_batch_size, train_val_ratio, seed, num_workers):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_val_ratio = train_val_ratio
        self.seed = seed
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # NOTE ((0.1307,), (0.3081,))，均值是0.1307，标准差是0.3081，由MNIST数据集提供方计算好的
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        pass
        # 已经有了就不用下载了
        # MNIST(self.data_dir, train=True, download=False)
        # MNIST(self.data_dir, train=False, download=False)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        # 分为fit阶段和test阶段

        if stage == 'fit' or stage is None:
            # 载入train数据集
            mnist_train = MNIST(self.dataset_dir, train=True, download=False, transform=self.transform)
            # 划分train数据集的train和val比例
            mnist_train_length = len(mnist_train)
            train_val = [int(mnist_train_length * ratio) for ratio in self.train_val_ratio]
            # 设置seed
            generator = torch.Generator().manual_seed(self.seed)

            self.mnist_train, self.mnist_val = random_split(mnist_train, train_val, generator=generator)

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.dataset_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.test_batch_size, num_workers=self.num_workers)
