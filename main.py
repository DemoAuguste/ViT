import torch
import torch.nn as nn

from models.func import get_models
from utils import get_args
from utils.utils import adjust_learning_rate, save_checkpoint
from utils.dataloader import get_dataloader
from utils.train import train, validate



if __name__ == '__main__':
    args = get_args()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    ngpus_per_node = torch.cuda.device_count()
    train_loader, val_loader = get_dataloader(args=args)
    model = get_models(args=args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    best_acc1 = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

    

    
    



