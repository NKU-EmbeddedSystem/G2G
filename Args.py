import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Total
    parser.add_argument('--algorithm', type=str, default='fed_mutual',
                        help='Type of algorithms:{fed_mutual, fed_avg, normal}')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument('--node_num', type=int, default=3,
                        help='Number of nodes')
    parser.add_argument('--R', type=int, default=50,
                        help='Number of rounds: R')
    parser.add_argument('--E', type=int, default=7,
                        help='Number of local epochs: E')
    parser.add_argument('--notes', type=str, default='',
                        help='Notes of Experiments')
    parser.add_argument('--pin', type=bool, default=True,
                        help='pin-memory')

    # Model
    parser.add_argument('--global_model', type=str, default='ResNet50',
                        help='Type of global model: {LeNet5, MLP, CNN2, ResNet50,ResNet18,VGG16,Alexnet,Alexnet2}')
    parser.add_argument('--local_model', type=str, default='ResNet50',
                        help='Type of local model: {LeNet5, MLP, CNN2, ResNet50,ResNet18,VGG16,Alexnet,Alexnet2}')
    parser.add_argument('--catfish', type=str, default=None,
                        help='Type of local model: {None, LeNet5, MLP, CNN2, ResNet50}')

    # Data
    parser.add_argument('--dataset', type=str, default='pacs',
                        help='datasets: {air_ori, air, pacs, cifar100, cifar10, femnist,office-home, mnist}')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--split', type=int, default=5,
                        help='data split')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='val_ratio')
    parser.add_argument('--all_data', type=bool, default=True,
                        help='use all train_set')
    parser.add_argument('--classes', type=int, default=7,
                        help='classes')

    # Optima
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer: {sgd, adam}')
    parser.add_argument('--lr', type=float, default=0.0008,
                        help='learning rate')
    parser.add_argument('--lr_step', type=int, default=10,
                        help='learning rate decay step size')
    parser.add_argument('--stop_decay', type=int, default=50,
                        help='round when learning rate stop decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='local ratio of data loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='meme ratio of data loss')
    parser.add_argument('--workers', type=int, default=16,
                        help='num_workers')                    
    parser.add_argument('--pretrained',type=bool,
                        default=True)
    parser.add_argument('--factor', type=float, default=0.1, 
                        help='lr decreased factor (0.1)')
    parser.add_argument('--patience', type=int, default=20, 
                        help='number of epochs to want before reduce lr (20)')
    parser.add_argument('--lr-threshold', type=float, default=1e-4, 
                        help='lr schedular threshold')
    parser.add_argument('--ite-warmup', type=int, default=100, 
                        help='LR warm-up iterations (default:500)')
    # parser.add_argument('--label_smoothing', type=float, default=0.1, 
    #                     help='the rate of wrong label(default:0.2)')

    # for ALexnet2
    parser.add_argument('--lr0', type=float, default=0.0001, help='learning rate 0')
    parser.add_argument('--lr1', type=float, default=0.0007, help='learning rate 1')
    parser.add_argument('--weight-dec', type=float, default=1e-7, help='0.005 weight decay coefficient default 1e-5')
    parser.add_argument('--rp-size', type=int, default=1024, help='Random Projection size 1024')
    parser.add_argument('--hidden_size', type=int, default=4096, help='the size of hidden feature')
    parser.add_argument('--iteration', type=int, default=0, help='the iteration')

    parser.add_argument('--mix', type=float, default=0)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--warm_up',type=bool, default=True)
    parser.add_argument('--lr_scheduler',type=bool, default=True)
    args = parser.parse_args()
    return args
