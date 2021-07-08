import torch
import torchvision.models as models
# from vit_pytorch import ViT
from pytorch_pretrained_vit import ViT

"""
['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'googlenet', 'inception_v3', 
'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'resnet101', 'resnet152', 
'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 
'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 
'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 
'wide_resnet50_2']
"""


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_models(args):
    global model_names
    if args.arch in model_names:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()
    else:
        if args.arch.lower() == "vit": # add vision transformer
            if args.pretrained:
                print("=> using pre-trained model '{}'".format(args.arch))
                model = ViT('B_16_imagenet1k', pretrained=True, num_classes=args.num_classes)
            else:
                print("=> creating model '{}'".format(args.arch))
                model = ViT('B_16_imagenet1k', pretrained=False, num_classes=args.num_classes)

            # model = ViT(image_size=args.image_size,
            #             patch_size=args.patch_size,
            #             num_classes=args.num_classes,
            #             dim=1024, # use default settings
            #             depth=6,
            #             heads=16,
            #             mlp_dim=2048,
            #             dropout=0.1,
            #             emb_dropout=0.1)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    return model
        


if __name__ == '__main__':
    print(model_names)
    # print('test')