import copy
import torch
import Model
from torch import optim
from utils import LabelSmoothingLoss, GradualWarmupScheduler


def init_model(model_type,args):
    model = []
    if model_type == 'LeNet5':
        model = Model.LeNet5()
    elif model_type == 'MLP':
        model = Model.MLP()
    elif model_type == 'ResNet50':
        model = Model.ResNet50(args)
    elif model_type == 'ResNet18':
        model = Model.ResNet18(args)
    elif model_type == 'VGG16':
        model = Model.VGG16(args)
    elif model_type == 'Alexnet':
        model = Model.Alexnet(args)
    elif model_type == 'CNN':
        model = Model.CNN()
    return model


def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


class Node(object):
    def __init__(self, num, train_data, test_data, args):
        self.args = args
        self.num = num + 1
        self.device = self.args.device
        self.train_data = train_data
        self.test_data = test_data
        self.model = init_model(self.args.local_model,args).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)
        self.meme = init_model(self.args.global_model,args).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, self.args)
        self.Dict = self.meme.state_dict()

        afsche_local = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_local = GradualWarmupScheduler(self.optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_local)

        afsche_meme = optim.lr_scheduler.ReduceLROnPlateau(self.meme_optimizer, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_meme = GradualWarmupScheduler(self.meme_optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler= afsche_meme)

        # self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=args.classes)

    def fork(self, global_node):
        self.meme = copy.deepcopy(global_node.model).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, self.args)

    # def local_fork(self, global_node):
    #     self.model = copy.deepcopy(global_node.model).to(self.device)
    #     self.model_optimizer = init_optimizer(self.model, self.args)


class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        self.model = init_model(self.args.global_model,args).to(self.device)
        self.model_optimizer = init_optimizer(self.model, self.args)
        self.test_data = test_data
        self.Dict = self.model.state_dict()
        afsche_global = optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_global = GradualWarmupScheduler(self.model_optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler = afsche_global)

    def merge(self, Node_List):
        weights_zero(self.model)
        Node_State_List = [copy.deepcopy(Node_List[i].meme.state_dict()) for i in range(len(Node_List))]
        for key in self.Dict.keys():
            for i in range(len(Node_List)):
                self.Dict[key] += Node_State_List[i][key]
            self.Dict[key] = self.Dict[key]/len(Node_List)

    def fork(self, node):
        self.model = copy.deepcopy(node.meme).to(self.device)
        self.model_optimizer = init_optimizer(self.model, self.args)

    # def fork_local(self, node):
    #     self.model = copy.deepcopy(node.model).to(self.device)
    #     self.model_optimizer = init_optimizer(self.model, self.args)