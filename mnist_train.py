import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
import os
import socket
import time
import platform

def find_free_port():
    """Find a free port on the machine"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup_distributed(rank, world_size, master_addr, master_port):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    # For Windows compatibility
    if platform.system() == 'Windows':
        os.environ['GLOO_SOCKET_IFNAME'] = 'Ethernet'
    
    # Try multiple initialization methods
    init_methods = [
        lambda: dist.init_process_group(
            "gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        ),
        lambda: dist.init_process_group(
            "gloo",
            init_method=f"file://shared_file",
            rank=rank,
            world_size=world_size,
        ),
    ]
    
    last_exception = None
    for init_method in init_methods:
        try:
            init_method()
            print(f"Process {rank} initialized successfully")
            break
        except Exception as e:
            last_exception = e
            continue
    else:
        raise RuntimeError(f"Failed to initialize process group: {last_exception}")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        print(f"Process {rank} using GPU: {torch.cuda.get_device_name()}")
    else:
        print(f"Process {rank} using CPU")

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, epoch, rank):
    """Training function for one epoch"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Process {rank} | Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')

def main():
    parser = argparse.ArgumentParser(description='Distributed MNIST Training')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--master-addr', type=str, required=True)
    parser.add_argument('--master-port', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--checkpoint-dir', type=str, help='Directory to load checkpoint from')
    parser.add_argument('--start-checkpoint', type=str, help='Checkpoint to start from')
    args = parser.parse_args()

    # Create directories
    checkpoint_dir = os.path.join(args.data_dir, 'checkpoints')
    data_dir = os.path.join(args.data_dir, 'trainingSet')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Set up port
    try:
        port = int(args.master_port)
        if args.rank == 0:  # Only master tries to bind
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((args.master_addr, port))
    except OSError:
        if args.rank == 0:
            port = find_free_port()
            print(f"Port {args.master_port} is in use, using port {port} instead")
        else:
            port = int(args.master_port)
    
    # Set up distributed environment
    setup_distributed(args.rank, args.world_size, args.master_addr, port)
    
    # Setup data loading
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=args.rank == 0,
        transform=transform
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )

    # Set up model and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTNet().to(device)
    
    if args.world_size > 1:
        model = DDP(model)
        
    optimizer = optim.Adam(model.parameters())

    # Load checkpoint if specified
    start_epoch = 0
    if args.start_checkpoint and args.checkpoint_dir:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.start_checkpoint)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint from epoch {start_epoch}")
    
    if args.rank == 0:
        print(f"Starting training with {args.world_size} processes")
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Batch size per process: {train_loader.batch_size}")
        print(f"Training device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")

    # Training loop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_sampler.set_epoch(epoch)
        train(model, device, train_loader, optimizer, epoch, args.rank)
        
        if args.rank == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pt'))
        
        if args.world_size > 1:
            dist.barrier()
    
    if args.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
