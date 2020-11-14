from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
# custom module
from model import DGCNN_cls
from util import cal_loss, IOStream
from data import ScanObject_coseg, TrainingBatchSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4"


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.path):
        os.makedirs('checkpoints/'+args.path)
    if not os.path.exists('checkpoints/'+args.path+'/'+'models'):
        os.makedirs('checkpoints/'+args.path+'/'+'models')
    os.system('cp exp_classification.py checkpoints'+'/' +
              args.path+'/'+'exp_classification.py.backup')
    os.system('cp model.py checkpoints' + '/' +
              args.path + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' +
              args.path + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' +
              args.path + '/' + 'data.py.backup')


def train(args, io, model, train_loader, opt):
    # I will NEVER train the model on a CPU so I simply set "cuda"
    device = torch.device("cuda")

    criterion = cal_loss
    train_loss = 0.0
    count = 0.0
    model.train()
    train_pred = []
    train_true = []
    for data, label, mask in train_loader:
        # load data
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]

        # forward propagation and back propagation
        opt.zero_grad()
        logits = model(data)
        loss = criterion(logits, label)
        loss.backward()
        opt.step()

        # record results
        preds = logits.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

    # record results
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    io.cprint('Train  '
              'loss: %.6f, '
              'train acc: %.6f, '
              'train avg acc: %.6f'
              % (train_loss / count,
                 accuracy_score(train_true, train_pred),
                 balanced_accuracy_score(train_true, train_pred)))

    return


def test(args, io, model, test_loader, test_name,n_point):
    with torch.no_grad():
        device = torch.device("cuda")
        model = model.eval()

        # initialize the parameters
        criterion = cal_loss
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label, mask in test_loader:
            # load data
            data, label = data.to(device), label.to(device).squeeze()
            #Farest point sample
            #data: (B*N*3) => (B*n_point*3)
            data_flipped = data.transpose(1, 2).contiguous()
            data = (
                pointnet2_utils.gather_operation(
                data_flipped, pointnet2_utils.furthest_point_sample(data, n_point)
                )
                .transpose(1, 2)
                .contiguous()
                if n_point is not None
                else None
            )
     
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            # predict
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]

            # record results
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        # record results
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = accuracy_score(test_true, test_pred)
        avg_per_class_acc = balanced_accuracy_score(test_true, test_pred)

        io.cprint(' * %s '
                  'loss: %.6f, '
                  'test acc: %.6f, '
                  'test avg acc: %.6f'
                  % (test_name,
                     test_loss / count,
                     test_acc,
                     avg_per_class_acc))

    return test_acc


def experiment(n_points_choices, path):
    args.n_points_choices = n_points_choices
    args.path = path

    # make path
    _init_()

    # record args
    io = IOStream('checkpoints/' + args.path + '/run.log')
    io.cprint(str(args))

    io.cprint('Using GPU : ' + str(torch.cuda.current_device()) +
              ' from ' + str(torch.cuda.device_count()) + ' devices')

    # set seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # initialize train_loader
    train_dataset = ScanObject_coseg(partition='training')
    train_sampler = TrainingBatchSampler(train_dataset,
                                         args.n_points_choices,
                                         args.batch_size)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=32)

    # initialize test data loaders
    test_dataset256 = ScanObject_coseg(partition='test', n_points=256)
    test_loader256 = DataLoader(test_dataset256, num_workers=16,
                                batch_size=args.test_batch_size)
    test_dataset512 = ScanObject_coseg(partition='test', n_points=512)
    test_loader512 = DataLoader(test_dataset512, num_workers=16,
                                batch_size=args.test_batch_size)
    test_dataset1024 = ScanObject_coseg(partition='test', n_points=1024)
    test_loader1024 = DataLoader(test_dataset1024, num_workers=16,
                                 batch_size=args.test_batch_size)
    test_dataset2048 = ScanObject_coseg(partition='test', n_points=2048)
    test_loader2048 = DataLoader(test_dataset2048, num_workers=16,
                                 batch_size=args.test_batch_size)

    # Load models
    device = torch.device("cuda")
    model = DGCNN_cls(args).to(device)
    model = nn.DataParallel(model)

    # Use SGD and CosineAnnealingLR to train
    print("Use SGD")
    opt = optim.SGD(model.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)

    # start training
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    for i in range(args.epochs):
        io.cprint('Epoch [%d]' % (i + 1))
        # train model
        train(args, io, model, train_loader, opt)

        # adjust learning rate
        scheduler.step()

        # test
        test(args, io, model, test_loader256, 'Test 256 ',256)
        test(args, io, model, test_loader512, 'Test 512 ',515)
        test(args, io, model, test_loader1024, 'Test 1024',1024)
        test(args, io, model, test_loader2048, 'Test 2048',2048)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='scanobjectnn', metavar='N',
                        choices=['scanobjectnn'])
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    args = parser.parse_args()

    # experiment([256], 'train256')
    # experiment([512], 'train512')
    # experiment([1024], 'train1024')
    experiment([2048], 'train2048')
    experiment([256, 512, 1024, 2048], 'train_mix')
