""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), modified. """

import argparse
import os
import logging
import torch
import torch.utils.data
import torchvision

import data_utils
import trainer

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # io settings.
    parser.add_argument('--outfile', type=str, default='./logs/2021_04_17_train_modelnet',
                        metavar='BASENAME', help='output filename (prefix)')
    parser.add_argument('--dataset_path', type=str, default='./dataset/ModelNet',
                        metavar='PATH', help='path to the input dataset')
    
    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', type=str,
                        metavar='DATASET', help='dataset type')
    parser.add_argument('--data_type', default='synthetic', type=str,
                        metavar='DATASET', help='whether data is synthetic or real')
    parser.add_argument('--categoryfile', type=str, default='./dataset/modelnet40_half1.txt',
                        metavar='PATH', help='path to the categories to be trained')
    parser.add_argument('--num_points', default=1000, type=int,
                        metavar='N', help='points in point-cloud.')
    parser.add_argument('--num_random_points', default=100, type=int,
                        metavar='N', help='number of random points to compute Jacobian.')
    parser.add_argument('--mag', default=0.8, type=float,
                        metavar='D', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
    parser.add_argument('--sigma', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--clip', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--workers', default=12, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for Embedding
    parser.add_argument('--embedding', default='pointnet',
                        type=str, help='pointnet')
    parser.add_argument('--dim_k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector')
    
    # settings for LK
    parser.add_argument('--max_iter', default=10, type=int,
                        metavar='N', help='max-iter on LK.')

    # settings for training.
    parser.add_argument('--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--max_epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='manual epoch number')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        metavar='METHOD', help='name of an optimizer')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('--lr', type=float, default=1e-3,
                        metavar='D', help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, 
                        metavar='D', help='decay rate of learning rate')

    # settings for log
    parser.add_argument('--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file')

    args = parser.parse_args(argv)
    return args


def train(args, trainset, evalset, dptnetlk):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    model = dptnetlk.create_model()

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

    model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
    print('resume epoch from {}'.format(args.start_epoch+1))

    evalloader = torch.utils.data.DataLoader(evalset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    min_loss = float('inf')
    min_info = float('inf')

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=args.lr, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=args.lr)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        min_info = checkpoint['min_info']
        optimizer.load_state_dict(checkpoint['optimizer'])

    # training
    LOGGER.debug('Begin Training!')
    for epoch in range(args.start_epoch, args.max_epochs):
        running_loss, running_info = dptnetlk.train_one_epoch(
            model, trainloader, optimizer, args.device, 'train', args.data_type, num_random_points=args.num_random_points)
        val_loss, val_info = dptnetlk.eval_one_epoch(
            model, evalloader, args.device, 'eval', args.data_type, num_random_points=args.num_random_points)
        
        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1,
                    running_loss, val_loss, running_info, val_info)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'min_info': min_info,
                'optimizer': optimizer.state_dict(), }
        if is_best:
            torch.save(model.state_dict(), '{}_{}.pth'.format(args.outfile, 'model_best'))
        torch.save(snap, '{}_{}.pth'.format(args.outfile, 'snap_last'))


def main(args):
    trainset, evalset = get_datasets(args)
    dptnetlk = trainer.TrainerAnalyticalPointNetLK(args)
    train(args, trainset, evalset, dptnetlk)


def get_datasets(args):
    cinfo = None
    if args.categoryfile:
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                    data_utils.Mesh2Points(),\
                    data_utils.OnUnitCube(),\
                    data_utils.Resampler(args.num_points)])

        traindata = data_utils.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        evaldata = data_utils.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        trainset = data_utils.PointRegistration(traindata, data_utils.RandomTransformSE3(args.mag))
        evalset = data_utils.PointRegistration(evaldata, data_utils.RandomTransformSE3(args.mag))
    else:
        print('wrong dataset type!')

    return trainset, evalset


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Training (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)

    LOGGER.debug('Training completed! Yay~~ (PID=%d)', os.getpid())
