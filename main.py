import torch

from utils import get_args


if __name__ == '__main__':
    args = get_args()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    
    



