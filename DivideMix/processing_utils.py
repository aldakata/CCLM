import torch
import torch.optim as optim
import sys


def load_net_optimizer_from_ckpt_to_device(net, args, ckpt_path, device):
    print(f"[ LOADING NET] {ckpt_path}\n")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device(device)
    net.load_state_dict(ckpt["net"])
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    return net, optimizer


def save_net_optimizer_to_ckpt(net, optimizer, ckpt_path):
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
