import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


# ---------------------------
# 1. Setup distributed process
# ---------------------------
def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


# ---------------------------
# 2. Simple MNIST Model
# ---------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------
# 3. Training Function
# ---------------------------
def main():
    setup()

    rank = dist.get_rank()
    device = torch.device("cuda")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    sampler = DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler
    )

    # Model
    model = Net().to(device)
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(20):
        sampler.set_epoch(epoch)
        total_loss = 0

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Rank {rank}, Epoch {epoch}, Loss {total_loss:.4f}")

    cleanup()


if __name__ == "__main__":
    main()