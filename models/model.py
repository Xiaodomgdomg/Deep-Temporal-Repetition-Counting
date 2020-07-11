import torch
from torch import nn

from models import resnet, resnext


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnext'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(opt=opt)
        elif opt.model_depth == 18:
            model = resnet.resnet18(opt=opt)
        elif opt.model_depth == 34:
            model = resnet.resnet34(opt=opt)
        elif opt.model_depth == 50:
            model = resnet.resnet50(opt=opt)
        elif opt.model_depth == 101:
            model = resnet.resnet101(opt=opt)
        elif opt.model_depth == 152:
            model = resnet.resnet152(opt=opt)
        elif opt.model_depth == 200:
            model = resnet.resnet200(opt=opt)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(opt=opt)
        elif opt.model_depth == 101:
            model = resnext.resnet101(opt=opt)
        elif opt.model_depth == 152:
            model = resnext.resnet152(opt=opt)
   

    if not opt.no_cuda:
        model = model.cuda()




    return model, model.parameters()
