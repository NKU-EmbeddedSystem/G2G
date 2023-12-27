import torch
import h5py
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Data(object):
    def __init__(self, args,logger):
        self.args = args
        iteration = args.iteration
        client = None
        if args.dataset == 'pacs':
            self.trainset, self.testset = None, None
            if iteration == 0:
                client = ['cartoon', 'sketch',  'art_painting','photo']
                self.train_loader, self.test_loader,self.target_loader = get_pacs_loaders(args,client)
            if iteration == 1:
                client=[ 'photo','cartoon','sketch','art_painting']
                self.train_loader, self.test_loader,self.target_loader = get_pacs_loaders(args,client)
            if iteration == 2:
                client=[ 'sketch','photo','art_painting','cartoon']
                self.train_loader, self.test_loader,self.target_loader = get_pacs_loaders(args,client)
            
            if iteration == 3:
                client=[ 'photo','cartoon','art_painting','sketch']
                self.train_loader, self.test_loader,self.target_loader = get_pacs_loaders(args,client) 

        if args.dataset == 'vlcs':
            if iteration == 0:
                self.train_loader, self.test_loader,self.target_loader = get_vlcs_loaders(args,
                                                        client=['SUN09', 'Caltech101','LabelMe','VOC2007'])
            if iteration == 1:
                self.train_loader, self.test_loader,self.target_loader = get_vlcs_loaders(args,
                                                        client=['Caltech101','VOC2007','SUN09', 'LabelMe'])
            if iteration == 2:
                self.train_loader, self.test_loader,self.target_loader = get_vlcs_loaders(args,
                                                        client=['SUN09', 'LabelMe','VOC2007','Caltech101'])  
            if iteration == 3:
                self.train_loader, self.test_loader,self.target_loader = get_vlcs_loaders(args,
                                                        client=['LabelMe','Caltech101','VOC2007','SUN09'])
                                                               
            
            
        if args.dataset == 'office-home':
            if iteration == 0:
                self.train_loader, self.test_loader,self.target_loader = get_office_loaders(args,                                                     
                                                        client=['Real World','Product','Clipart','Art'])                                                        
            if iteration == 1:
                self.train_loader, self.test_loader,self.target_loader = get_office_loaders(args,
                                                        client=[ 'Real World','Product','Art','Clipart'])
                                                       
            if iteration == 2:
                self.train_loader, self.test_loader,self.target_loader = get_office_loaders(args,
                                                        client=['Real World','Art','Clipart', 'Product'])                             
            if iteration == 3:
                self.train_loader, self.test_loader,self.target_loader = get_office_loaders(args,
                                                        client=['Clipart', 'Art','Product','Real World'])

        logger.info('CLIENT_ORDER{}'.format(client))



class Loader_dataset(data.Dataset):
    def __init__(self, path, tranforms = None):
        self.path = path
        self.dataset = datasets.ImageFolder(path, transform=tranforms)
        self.length = self.dataset.__len__()
        self.transform = tranforms

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data, label = self.dataset.__getitem__(idx)
        return data, label

def get_vlcs_loaders(args, client):
    path_root = 'datasets/VLCS/'
    trans0 = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([225, 225]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    for i in range(3):
        train_path[i] = path_root + client[i] + '/train'
        train_datas[i] = Loader_dataset(path=train_path[i], tranforms=trans0)
        train_loaders[i] = DataLoader(train_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)

        valid_path[i] = path_root + client[i] + '/val'
        valid_datas[i] = Loader_dataset(path=valid_path[i], tranforms=trans1)
        valid_loaders[i] = DataLoader(valid_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    target_path = path_root + client[3] + '/val'
    target_data = Loader_dataset(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    print(client,'\n')
    return train_loaders, valid_loaders, target_loader

class Loader_dataset_pacs(data.Dataset):
    def __init__(self, path, tranforms):
        self.path = path
        hdf = h5py.File(self.path, 'r')
        self.length = len(hdf['labels'])   # <KeysViewHDF5 ['images', 'labels']>
        self.transform = tranforms
        hdf.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hdf = h5py.File(self.path, 'r')
        y = hdf['labels'][idx]
        data_pil = Image.fromarray(hdf['images'][idx, :, :, :].astype('uint8'), 'RGB')
        hdf.close()
        data = self.transform(data_pil)
        return data, torch.tensor(y).long().squeeze()-1

def get_pacs_loaders(args, client = ['cartoon', 'sketch',  'art_painting','photo']):
    path_root = 'datasets/PACS/'
    trans0 = transforms.Compose([transforms.RandomResizedCrop(222, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([222, 222]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    for i in range(3):
        train_path[i] = path_root + client[i] + '_train.hdf5'
        train_datas[i] = Loader_dataset_pacs(path=train_path[i], tranforms=trans0)
        train_loaders[i] = DataLoader(train_datas[i], args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
        valid_path[i] = path_root + client[i] + '_val.hdf5'
        valid_datas[i] = Loader_dataset_pacs(path=valid_path[i], tranforms=trans1)
        valid_loaders[i] = DataLoader(valid_datas[i], args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
    target_path = path_root + client[3] + '_test.hdf5'
    target_data = Loader_dataset_pacs(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
    return train_loaders, valid_loaders, target_loader


def get_office_loaders(args, client):
    path_root = 'datasets/OfficeHome/'
    trans0 = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([225, 225]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    for i in range(3):
        train_path[i] = path_root + client[i] + '/train'
        train_datas[i] = Loader_dataset(path=train_path[i], tranforms=trans0)
        train_loaders[i] = DataLoader(train_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)

        valid_path[i] = path_root + client[i] + '/val'
        valid_datas[i] = Loader_dataset(path=valid_path[i], tranforms=trans1)
        valid_loaders[i] = DataLoader(valid_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    target_path = path_root + client[3] + '/val'
    target_data = Loader_dataset(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    return train_loaders, valid_loaders, target_loader  