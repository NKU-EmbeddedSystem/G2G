import torch
import Node
from torch.optim.lr_scheduler import _LRScheduler,ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):

    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, init_lr=1e-7, after_scheduler=None):
        self.init_lr = init_lr
        assert init_lr > 0, 'Initial LR should be greater than 0.'
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs

        return [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if (self.finished and self.after_scheduler) or self.total_epoch == 0:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class Recorder(object):
    def __init__(self, args, logger):
        self.args = args
        self.counter = 0
        self.tra_loss = {}
        self.tra_acc = {}
        self.val_loss = {}
        self.val_acc = {}
        self.logger = logger
        for i in range(self.args.node_num + 1):
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
        self.acc_best = torch.zeros(self.args.node_num + 1)
        self.get_a_better = torch.zeros(self.args.node_num + 1)

    def validate(self, node):
        self.counter += 1
        node.model.to(node.device).eval()
        total_loss = 0.0
        correct = 0.0

        with torch.no_grad():
            for idx, (data, target) in enumerate(node.test_data):
                data, target = data.to(node.device), target.to(node.device)
                output = node.model(data)
                total_loss += torch.nn.CrossEntropyLoss()(output, target)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss = total_loss / (idx + 1)
            acc = correct / len(node.test_data.dataset) * 100
        self.val_loss[str(node.num)].append(total_loss)
        self.val_acc[str(node.num)].append(acc)

        if self.val_acc[str(node.num)][-1] > self.acc_best[node.num]:
            self.get_a_better[node.num] = 1
            self.acc_best[node.num] = self.val_acc[str(node.num)][-1]
            torch.save(node.model.state_dict(),
                       'save/model/Node{:d}_{:s}.pt'.format(node.num, node.args.local_model))
            # add warm_up lr 
            if self.args.warm_up == True and str(node.num) != '0':
                node.sche_local.step(metrics=self.val_acc[str(node.num)][-1])
                node.sche_meme.step(metrics=self.val_acc[str(node.num)][-1])

        if self.val_acc[str(node.num)][-1] <= self.acc_best[node.num]:
            print('##### Node{:d}: Not better Accuracy: {:.2f}%'.format(node.num, self.val_acc[str(node.num)][-1]))


        node.meme.to(node.device).eval()
        total_loss = 0.0
        correct = 0.0

        with torch.no_grad():
            for idx, (data, target) in enumerate(node.test_data):
                data, target = data.to(node.device), target.to(node.device)
                output = node.meme(data)
                total_loss += torch.nn.CrossEntropyLoss()(output, target)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss = total_loss / (idx + 1)
            acc = correct / len(node.test_data.dataset) * 100
      
    def log(self, node):
        return self.val_acc[str(node.num)][-1], self.val_loss[str(node.num)][-1]

    def printer(self, node):
        if self.get_a_better[node.num] == 1 and node.num == 0:
            print('Node{:d}: A Better Accuracy: {:.2f}%! Model Saved!'.format(node.num, self.acc_best[node.num]))
            self.get_a_better[node.num] = 0
        elif self.get_a_better[node.num] == 1:
            self.get_a_better[node.num] = 0

    def finish(self):
        torch.save([self.val_loss, self.val_acc],
                   'save/record/loss_acc_{:s}_{:s}.pt'.format(self.args.algorithm, self.args.notes))
        print('Finished!\n')
        for i in range(self.args.node_num + 1):
            print('Node{}: Best Accuracy = {:.2f}%'.format(i, self.acc_best[i]))


def Catfish(Node_List, args):
    if args.catfish is None:
        pass
    else:
        Node_List[0].model = Node.init_model(args.catfish)
        Node_List[0].optimizer = Node.init_optimizer(Node_List[0].model, args)


def LR_scheduler(rounds, Node_List, args, Global_node = None):
    #     trigger = 7
    if rounds > 15 and rounds <=30:
        trigger = 15
    elif rounds > 30 and rounds <=45:
        trigger = 25
    elif rounds > 45 and rounds <=50:
        trigger = 40
    else:
        trigger = 51

    if rounds != 0 and rounds % trigger == 0 and rounds < args.stop_decay:
        args.lr *= 0.5
        for i in range(len(Node_List)):
            Node_List[i].args.lr = args.lr
            Node_List[i].args.alpha = args.alpha
            Node_List[i].args.beta = args.beta
            Node_List[i].optimizer.param_groups[0]['lr'] = args.lr
            Node_List[i].meme_optimizer.param_groups[0]['lr'] = args.lr
        if Global_node !=None:
            Global_node.args.lr = args.lr
            Global_node.model_optimizer.param_groups[0]['lr'] = args.lr
    print('Learning rate={:.10f}'.format(args.lr))


def Summary(args):
    print("Summary:\n")
    print("algorithm:{}\n".format(args.algorithm))
    print("dataset:{}\tbatchsize:{}\n".format(args.dataset, args.batch_size))
    print("node_num:{},\tsplit:{}\n".format(args.node_num, args.split))
    # print("iid:{},\tequal:{},\n".format(args.iid == 1, args.unequal == 0))
    print("global epochs:{},\tlocal epochs:{},\n".format(args.R, args.E))
    print("global_model:{},\tlocal model:{},\n".format(args.global_model, args.local_model))