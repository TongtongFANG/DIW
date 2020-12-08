import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import *
from kmm import *
import dataloader


parser = argparse.ArgumentParser()
parser.add_argument('--noise_type', type=str, default='symmetric', help='label noise type, either pair or symmetric')
parser.add_argument('--noise_rate', type=float, default=0.4, help='label noise rate, should be less than 1')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--step', type=float, default=100, help='period of learning rate decay')
parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--num_val', type=int, default=1000, help='total number of validation data')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for training data')
parser.add_argument('--batch_size_val', type=int, default=256, help='batch size for validation data')
parser.add_argument('--num_epoch', type=int, default=400, help='total number of training epoch')
parser.add_argument('--seed', type=int, default=100, help='random seed')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ['PYTHONHASHSEED'] = '0'
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def build_model():
    net = LeNet(n_out=10)

    if torch.cuda.is_available():
        net.cuda()

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=args.gamma)

    return net, opt, scheduler


def main():
    # data loaders
    train_loader, val_loader, test_loader = dataloader.F_MNIST_dataloader(args.batch_size, args.batch_size_val,
                                                                       args.noise_type,
                                                                       args.noise_rate, args.num_val).run()

    # define the model, optimizer, and lr decay scheduler
    net, opt, scheduler = build_model()

    # train the model
    test_acc = []

    for epoch in tqdm(range(args.num_epoch)):

        train_acc_tmp = []
        test_acc_tmp = []

        for i, (image, labels, _) in enumerate(train_loader):

            # weight estimation (we) step
            net.eval()

            image = to_cuda(image)
            labels = to_cuda(labels)

            out_train = net(image)
            l_tr = F.cross_entropy(out_train, labels, reduction='none')

            val_image, val_labels, val__ = next(iter(val_loader))

            val_image = to_cuda(val_image)
            val_labels = to_cuda(val_labels)

            out_val = net(val_image)

            l_val = F.cross_entropy(out_val, val_labels, reduction='none')
            l_tr_reshape = np.array(l_tr.detach().cpu()).reshape(-1, 1)
            l_val_reshape = np.array(l_val.detach().cpu()).reshape(-1, 1)

            # warm start
            if epoch < 1:
                coef = [1 for i in range(len(_))]
            # obtain importance weights
            else:
                kernel_width = get_kernel_width(l_tr_reshape)
                coef = kmm(l_tr_reshape, l_val_reshape, kernel_width)

            w = torch.from_numpy(np.asarray(coef)).float().cuda()

            # weighted classification (wc) step
            net.train()
            out_train_wc = net(image)
            l_tr_wc = F.cross_entropy(out_train_wc, labels, reduction='none')
            l_tr_wc_weighted = torch.sum(l_tr_wc * w)

            opt.zero_grad()
            l_tr_wc_weighted.backward()
            opt.step()

            # train acc
            train_correct = 0
            train_total = 0
            _, predicted = torch.max(out_train_wc.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_accuracy = train_correct / train_total
            train_acc_tmp.append(train_accuracy)

        train_accuracy_mean = np.mean(train_acc_tmp)
        print("train accuracy mean is", train_accuracy_mean)

        net.eval()
        # test acc
        for itr, (test_img, test_label, __) in enumerate(test_loader):
            test_img = to_cuda(test_img)
            test_label = to_cuda(test_label)
            test_correct = 0
            test_total = 0
            out_test = net(test_img)
            _, predicted = torch.max(out_test.data, 1)
            test_total += test_label.size(0)
            test_correct += (predicted == test_label).sum().item()
            test_accuracy = test_correct / test_total

            test_acc_tmp.append(test_accuracy)

        test_accuracy_mean = np.mean(test_acc_tmp)
        print("test accuracy mean is", test_accuracy_mean)
        test_acc.append(test_accuracy_mean)
        test_acc_arr = np.array(test_acc)

        scheduler.step()

    # save the output
    np.savetxt('./output/test_acc.txt', test_acc_arr, fmt='%s')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(test_acc)
    fig.savefig('./output/test_acc.png')


if __name__ == '__main__':
    main()
