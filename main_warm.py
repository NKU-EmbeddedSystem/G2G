import torch
from Node import Node, Global_Node
from Args import args_parser
from Data import Data
from utils import LR_scheduler, Recorder, Catfish, Summary
from Trainer import Trainer
from log import logger_config,set_random_seed

# init args
args = args_parser()
logger = logger_config(log_path='../logger/{}_{}_{}_alpha_{}_beta_{}.txt'.format(
                        args.lr,
                        args.E, 
                        args.batch_size,
                        args.alpha,
                        args.beta), 
                        logging_name='FedRD')
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('Running on', args.device)
Data = Data(args,logger)


# init nodes
Global_node = Global_Node(Data.target_loader, args)
Node_List = [Node(k, Data.train_loader[k], Data.test_loader[k], args) for k in range(args.node_num)]
Catfish(Node_List, args)

recorder = Recorder(args,logger)
Summary(args)
# start

for rounds in range(args.R):
    print('===============The {:d}-th round==============='.format(rounds + 1))
    if args.lr_scheduler == True:
        LR_scheduler(rounds, Node_List, args)
    Train = Trainer(args)
    for k in range(len(Node_List)):
        Node_List[k].fork(Global_node)
        for epoch in range(args.E):
            Train(Node_List[k],args,logger)
        recorder.printer(Node_List[k])
        Global_node.fork(Node_List[k])
        recorder.printer(Global_node)
        recorder.validate(Node_List[k])
    logger.info("iteration:{},epoch:{},accurancy:{},loss:{}".format(args.iteration,rounds,recorder.log(Global_node)[0],recorder.log(Global_node)[1]))
recorder.finish()
Summary(args)
