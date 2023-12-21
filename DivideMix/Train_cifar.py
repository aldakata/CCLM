from __future__ import print_function
import sys
import this
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from class_conditional_utils import ccgmm_codivide, gmm_codivide
import dataloader_cifar as dataloader

from processing_utils import (
    save_net_optimizer_to_ckpt,
    load_net_optimizer_from_ckpt_to_device,
)

parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
parser.add_argument("--batch_size", default=64, type=int, help="train batchsize")
parser.add_argument(
    "--lr", "--learning_rate", default=0.02, type=float, help="initial learning rate"
)
parser.add_argument("--noise_mode", default="sym")
parser.add_argument("--alpha", default=4, type=float, help="parameter for Beta")
parser.add_argument(
    "--lambda_u", default=25, type=float, help="weight for unsupervised loss"
)
parser.add_argument(
    "--p_threshold", default=0.5, type=float, help="clean probability threshold"
)
parser.add_argument("--T", default=0.5, type=float, help="sharpening temperature")
parser.add_argument("--num_epochs", default=300, type=int)
parser.add_argument("--r", default=0.5, type=float, help="noise ratio")
parser.add_argument("--id", default="")
parser.add_argument("--seed", default=123)
parser.add_argument("--gpuid", default=0, type=int)
parser.add_argument("--num_class", default=10, type=int)
parser.add_argument(
    "--data_path", default="./cifar-10", type=str, help="path to dataset"
)
parser.add_argument("--natural", type=str, help="path to natural noise")
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--resume", default=0, type=int)

parser.add_argument("--cc", default=False, dest="cc", action="store_true")
parser.set_defaults(cc=False)

parser.add_argument(
    "--codivide-log", default=False, dest="codivide_log", action="store_true"
)
parser.set_defaults(codivide_log=False)


parser.add_argument(
    "--skip-warmup", default=False, dest="skip_warmup", action="store_true"
)
parser.set_defaults(skip_warmup=False)

parser.add_argument("--confusion", default=False, dest="confusion", action="store_true")
parser.set_defaults(confusion=False)

parser.add_argument("--noise-type", dest="noise_type", default=None)

parser.add_argument("--workers", dest="workers", default=5, type=int)


args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = "cuda:0"


# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(
        labeled_trainloader
    ):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(
            1, labels_x.view(-1, 1), 1
        )
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = (
            inputs_x.cuda(),
            inputs_x2.cuda(),
            labels_x.cuda(),
            w_x.cuda(),
        )
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (
                torch.softmax(outputs_u11, dim=1)
                + torch.softmax(outputs_u12, dim=1)
                + torch.softmax(outputs_u21, dim=1)
                + torch.softmax(outputs_u22, dim=1)
            ) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (
                torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)
            ) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        logits_x = logits[: batch_size * 2]
        logits_u = logits[batch_size * 2 :]

        Lx, Lu, lamb = criterion(
            logits_x,
            mixed_target[: batch_size * 2],
            logits_u,
            mixed_target[batch_size * 2 :],
            epoch + batch_idx / num_iter,
            warm_up,
        )

        # regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write("\r")
        sys.stdout.write(
            "%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f"
            % (
                args.dataset,
                args.r,
                args.noise_mode,
                epoch,
                args.num_epochs,
                batch_idx + 1,
                num_iter,
                Lx.item(),
                Lu.item(),
            )
        )
        sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        if (
            args.noise_mode == "asym"
        ):  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty
        elif args.noise_mode == "sym":
            L = loss
        L.backward()
        optimizer.step()

        sys.stdout.write("\r")
        sys.stdout.write(
            "%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f"
            % (
                args.dataset,
                args.r,
                args.noise_mode,
                epoch,
                args.num_epochs,
                batch_idx + 1,
                num_iter,
                loss.item(),
            )
        )
        sys.stdout.flush()


def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    per_class_accuracy = np.zeros(args.num_class)
    total_predicted = torch.zeros(10000, device=device)
    total_GT = torch.zeros(10000, device=device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            toiter = predicted.cpu().numpy().tolist()
            for c in toiter:
                per_class_accuracy[c] += sum(predicted[targets == c] == c)
            for i, e in enumerate(toiter):
                pos = batch_idx * len(toiter)
                total_predicted[pos + i] = e
                total_GT[pos + i] = targets[i]
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    total_predicted = total_predicted.cpu().detach().numpy()
    total_GT = total_GT.cpu().detach().numpy()
    # cm = confusion_matrix(total_GT, total_predicted)

    acc = 100.0 * correct / total
    per_class_accuracy /= total / args.num_class
    std = per_class_accuracy.std()
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\t STD:%.2f%%\n" % (epoch, acc, std))
    test_log.write("Epoch:%d   Accuracy:%.2f\t STD:%.2f\n" % (epoch, acc, std))
    test_log.flush()
    return acc


def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(50000)
    per_class_accuracy = np.zeros(args.num_class)
    targets_all = torch.zeros(50000, device=device)
    targets_clean_all = torch.zeros(50000, device=device)

    with torch.no_grad():
        for _, (inputs, targets, clean_targets, index) in enumerate(
            eval_loader
        ):
            inputs, targets, clean_targets = (
                inputs.cuda(),
                targets.cuda(),
                clean_targets.cuda(),
            )
            outputs = model(inputs)
            loss = CE(outputs, targets)
            _, predicted = torch.max(outputs, 1)
            for c in set(predicted.cpu().numpy()):
                per_class_accuracy[c] += sum(predicted[clean_targets == c] == c)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                targets_all[index[b]] = targets[b]
                targets_clean_all[index[b]] = clean_targets[b]

    targets_all = targets_all.cpu().numpy().astype("int")
    targets_clean_all = targets_clean_all.cpu().numpy().astype("int")

    losses = (losses - losses.min()) / (losses.max() - losses.min())

    all_loss.append(losses)

    if (
        args.r == 0.9
    ):  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)
    
    if args.cc:
        prob = ccgmm_codivide(input_loss, targets_all)
    else:
        prob = gmm_codivide(input_loss)

    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model


cc_sufix = "reg"
if args.cc:
    cc_sufix = "cc"

log_name = f"{args.dataset}_{args.r}_{args.noise_mode}_{args.p_threshold}_{cc_sufix}"

stats_log = open(
    f"./checkpoint/{log_name}_stats_resume.txt",
    "w",
)
test_log = open(
    f"./checkpoint/{log_name}_acc_resume.txt",
    "w",
)

train_log = open(
    f"./checkpoint/{log_name}_train_acc_resume.txt",
    "w",
)
memory_log = f"./checkpoint/{log_name}_memory"
os.makedirs(memory_log, exist_ok=True)

if args.dataset == "cifar10":
    warm_up = 10
elif args.dataset == "cifar100":
    warm_up = 30

if args.natural:
    noise_file = f"{args.natural}"
else:
    noise_file = "%s/%.1f_%s.json" % (args.data_path, args.r, args.noise_mode)

loader = dataloader.cifar_dataloader(
    args.dataset,
    r=args.r,
    noise_mode=args.noise_mode,
    batch_size=args.batch_size,
    num_workers=int(args.workers),
    root_dir=args.data_path,
    log=stats_log,
    noise_file=noise_file,
    noise_type=args.noise_type,
)

print("| Building net")
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True
resume_epoch = 0
criterion = SemiLoss()
CE = nn.CrossEntropyLoss(reduction="none")
CEloss = nn.CrossEntropyLoss()
if args.noise_mode == "asym":
    conf_penalty = NegEntropy()

if args.confusion:
    print("WARNING CONFUSION!\nLoading net")
    net1, optimizer1 = load_net_optimizer_from_ckpt_to_device(
        net1, args, f"./checkpoint/0.4_0.5_best_up_1.pt", device
    )
    net2, optimizer2 = load_net_optimizer_from_ckpt_to_device(
        net2, args, f"./checkpoint/0.4_0.5_best_up_2.pt", device
    )
    test_loader = loader.run("test")
    this_acc = test(0, net1, net2)


print("Loading net")
resume_epoch = 276
net1, optimizer1 = load_net_optimizer_from_ckpt_to_device(
    net1, args, f"./checkpoint/cifar100_0.9_sym_0.5_cc_memory/275_1.pt", device
)
net2, optimizer2 = load_net_optimizer_from_ckpt_to_device(
    net2, args, f"./checkpoint/cifar100_0.9_sym_0.5_cc_memory/275_2.pt", device
)


all_loss = [[], []]  # save the history of losses from two networks
best_acc = 0
for epoch in range(resume_epoch, args.num_epochs + 1):
    lr = args.lr
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group["lr"] = lr
    for param_group in optimizer2.param_groups:
        param_group["lr"] = lr
    test_loader = loader.run("test")
    eval_loader = loader.run("eval_train")

    if epoch < warm_up and not args.skip_warmup:
        warmup_trainloader = loader.run("warmup")
        print("Warmup Net1")
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print("\nWarmup Net2")
        warmup(epoch, net2, optimizer2, warmup_trainloader)

    else:
        prob1, all_loss[0] = eval_train(net1, all_loss[0])
        prob2, all_loss[1] = eval_train(net2, all_loss[1])

        pred1 = prob1 > args.p_threshold
        pred2 = prob2 > args.p_threshold

        print("Train Net1")
        labeled_trainloader, unlabeled_trainloader = loader.run(
            "train", pred2, prob2
        )  # co-divide
        train(
            epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader
        )  # train net1

        print("\nTrain Net2")
        labeled_trainloader, unlabeled_trainloader = loader.run(
            "train", pred1, prob1
        )  # co-divide
        train(
            epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader
        )  # train net2

    this_acc = test(epoch, net1, net2)
    if (not epoch % 25) or (epoch == warm_up) or epoch == 0:
        checkpoint1 = {
            "net": net1.state_dict(),
            "Model_number": 1,
            "Noise_Ratio": args.r,
            "Loss Function": "CrossEntropyLoss",
            "Optimizer": "SGD",
            "Noise_mode": args.noise_mode,
            "Accuracy": this_acc,
            "Dataset": args.dataset,
            "Batch Size": args.batch_size,
            "epoch": epoch,
        }
        checkpoint2 = {
            "net": net2.state_dict(),
            "Model_number": 2,
            "Noise_Ratio": args.r,
            "Loss Function": "CrossEntropyLoss",
            "Optimizer": "SGD",
            "Noise_mode": args.noise_mode,
            "Accuracy": this_acc,
            "Dataset": args.dataset,
            "Batch Size": args.batch_size,
            "epoch": epoch,
        }

        torch.save(checkpoint1, f"{memory_log}/{epoch}_1.pt")
        torch.save(checkpoint2, f"{memory_log}/{epoch}_2.pt")

    if this_acc > best_acc:
        best_acc = this_acc
        save_net_optimizer_to_ckpt(
            net1, optimizer1, f"./checkpoint/{args.r}_{args.p_threshold}_best_up_1.pt"
        )
        save_net_optimizer_to_ckpt(
            net2, optimizer2, f"./checkpoint/{args.r}_{args.p_threshold}_best_up_2.pt"
        )
