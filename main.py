import torch
import pytorch_lightning as pl
import os
import sys

# 引入自己的模块
sys.path.append(os.getcwd())

from a_mnist_dataset import MNISTDataModule
from b_teacher_and_student import TeacherNet, StudentNet
from c_pl_moudle import KDMoudle
from c_pl_moudle import get_callbacks


# %%
class Arguments:
    pass


# NOTE:数据集参数
dataset_args = Arguments()
dataset_args.seed = 42
dataset_args.Dataset_Dir = r"C:\files\2datasets"
dataset_args.train_batch_size = 64
dataset_args.test_batch_size = 1000
dataset_args.train_val_ratio = (0.8, 0.2)
dataset_args.num_workers = 0

# %%
# 实例化mnist数据集对象
mnist = MNISTDataModule(dataset_dir=dataset_args.Dataset_Dir,
                        train_batch_size=dataset_args.train_batch_size,
                        test_batch_size=dataset_args.test_batch_size,
                        train_val_ratio=dataset_args.train_val_ratio,
                        seed=dataset_args.seed,
                        num_workers=dataset_args.num_workers)
mnist.setup()

# 实例化dataloaders
train_dataloader = mnist.train_dataloader()
val_dataloader = mnist.val_dataloader()
test_dataloader = mnist.test_dataloader()

# %%
# NOTE:训练参数
train_args = Arguments()

train_args.learning_rate = 1e-3
train_args.max_epochs = 10
train_args.temperature = 2
train_args.alpha = 0.8
# %%
# 实例化pl_moudle
teacher = TeacherNet()
# 载入权重
teacher.load_state_dict(torch.load("./teacher_pretrain/mnist_cnn.pt"))
student = StudentNet()
kd_moudle = KDMoudle(teacher=teacher,
                     student=student,
                     learning_rate=train_args.learning_rate,
                     temperature=train_args.temperature,
                     alpha=train_args.alpha)
# %%
# 实例化trainer
trainer = pl.Trainer(
    # fast_dev_run=1,  # debug时开启，只跑一个batch的train、val和test
    max_epochs=train_args.max_epochs,
    callbacks=get_callbacks(),

    progress_bar_refresh_rate=20,
    flush_logs_every_n_steps=1,
    log_every_n_steps=1)

# %%
# training
trainer.fit(kd_moudle, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
# %%
# testing
trainer.test(dataloaders=test_dataloader)
