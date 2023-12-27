from ast import arg
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


def train_normal(node,args):
    node.model.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Training (the {:d}-batch): tra_Loss = {:.4f} tra_Accuracy = {:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(idx + 1, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100


def train_avg(node,args):
    node.meme.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.meme_optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.meme(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.meme_optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100
    node.model = node.meme


def train_mutual(node,args,logger):
    node.model.to(node.device).train()
    node.meme.to(node.device).train()
    train_loader = node.train_data
    total_local_loss = 0.0
    avg_local_loss = 0.0
    correct_local = 0.0
    acc_local = 0.0
    total_meme_loss = 0.0
    avg_meme_loss = 0.0
    correct_meme = 0.0
    acc_meme = 0.0
    train_index = 0
    total_global_kl_loss = 0.0
    total_local_kl_loss = 0.0
    avg_global_kl_loss = 0.0
    avg_local_kl_loss = 0.0
    description = 'Node{:d}: loss_model={:.4f} acc_model={:.2f}% loss_meme={:.4f} acc_meme={:.2f}%'
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            train_index = train_index + 1 
            node.optimizer.zero_grad()
            node.meme_optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_local_loss, acc_local, avg_meme_loss, acc_meme))
            data, target = data.to(node.device), target.to(node.device)
            output_local = node.model(data)
            output_meme = node.meme(data)

            kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))    
            kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))

            
            total_local_kl_loss += kl_local
            total_global_kl_loss += kl_meme
            
            _output_local = nn.Softmax(dim=1)(output_local)
            _, src_idx = torch.sort(_output_local, 1, descending=True)
            _output_meme = nn.Softmax(dim=1)(output_meme)
            _, src_idx_meme = torch.sort(_output_meme, 1, descending=True)
            if args.topk > 0:
                topk = np.min([args.topk, args.classes])
                for i in range( _output_local.size()[0]):
                    output_local[i, src_idx[i, topk:]] = (1.0 -  _output_local[i, src_idx[i, :topk]].sum())/ ( _output_local.size()[1] - topk)
                    output_meme[i, src_idx[i, topk:]] = (1.0 -  _output_meme[i, src_idx[i, :topk]].sum())/ ( _output_meme.size()[1] - topk)


            ce_local = CE_Loss(output_local, target)
            ce_meme = CE_Loss(output_meme, target)  

            loss_local = node.args.alpha * ce_local + (1 - node.args.alpha) * kl_local
            loss_meme = node.args.beta * ce_meme + (1 - node.args.beta) * kl_meme
            loss_local.backward()
            loss_meme.backward()
            node.optimizer.step()
            node.meme_optimizer.step()
            total_local_loss += loss_local
            avg_local_loss = total_local_loss / (idx + 1)
            pred_local = output_local.argmax(dim=1)
            correct_local += pred_local.eq(target.view_as(pred_local)).sum()
            acc_local = correct_local / len(train_loader.dataset) * 100
            total_meme_loss += loss_meme
            avg_meme_loss = total_meme_loss / (idx + 1)
            pred_meme = output_meme.argmax(dim=1)
            correct_meme += pred_meme.eq(target.view_as(pred_meme)).sum()
            acc_meme = correct_meme / len(train_loader.dataset) * 100  

            avg_global_kl_loss = total_global_kl_loss / (idx + 1)
            avg_local_kl_loss = total_local_kl_loss / (idx + 1)



            if args.mix > 0:
                mixed_total_local_loss = 0.0
                mixed_avg_local_loss = 0.0
                mixed_correct_local = 0.0
                mixed_acc_local = 0.0
                mixed_total_meme_loss = 0.0
                mixed_avg_meme_loss = 0.0
                mixed_correct_meme = 0.0
                mixed_acc_meme = 0.0

                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(data.size()[0]).cuda()

                other_data, other_target = data[index, :],target[index]
                other_data, other_target = other_data.to(node.device), other_target.to(node.device)

                mixed_input = lam * data + (1 - lam) * other_data
                mixed_label = lam * target + (1 - lam) * other_target

                mixed_output_local = node.model(mixed_input)
                mixed_output_meme = node.meme(mixed_input)

                # mixed_output_local,mixed_output_meme = mixed_output_local.to(node.device),mixed_output_meme.to(node.device)

                mixed_ce_local = CE_Loss(mixed_output_local, mixed_label.long())
                mixed_kl_local = KL_Loss(LogSoftmax(mixed_output_local), Softmax(mixed_output_meme.detach()))
                mixed_ce_meme = CE_Loss(mixed_output_meme, mixed_label.long())                    
                mixed_kl_meme = KL_Loss(LogSoftmax(mixed_output_meme), Softmax(mixed_output_local.detach()))
                mixed_loss_local = node.args.alpha * mixed_ce_local + (1 - node.args.alpha) * mixed_kl_local
                mixed_loss_meme = node.args.beta * mixed_ce_meme + (1 - node.args.beta) * mixed_kl_meme
                
                (args.mix * mixed_loss_local).backward()
                (args.mix * mixed_loss_meme).backward()
                node.optimizer.step()
                node.meme_optimizer.step()
                mixed_total_local_loss += mixed_loss_local
                mixed_avg_local_loss = mixed_total_local_loss / (idx + 1)
                mixed_pred_local = mixed_output_local.argmax(dim=1)
                mixed_correct_local += mixed_pred_local.eq(target.view_as(pred_local)).sum()
                mixed_acc_local = mixed_correct_local / len(train_loader.dataset) * 100
                mixed_total_meme_loss += mixed_loss_meme
                mixed_avg_meme_loss = mixed_total_meme_loss / (idx + 1)
                mixed_pred_meme = mixed_output_meme.argmax(dim=1)
                mixed_correct_meme += mixed_pred_meme.eq(mixed_label.view_as(mixed_pred_meme)).sum()
                mixed_acc_meme = mixed_correct_meme / len(train_loader.dataset) * 100      



class Trainer(object):

    def __init__(self, args, logger=None):
        if args.algorithm == 'fed_mutual':
            self.train = train_mutual
        elif args.algorithm == 'fed_avg':
            self.train = train_avg
        elif args.algorithm == 'normal':
            self.train = train_normal

    def __call__(self, node,args,logger):
        self.train(node,args,logger)
