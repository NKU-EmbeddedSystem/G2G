# G2G
This is the Official PyTorch implemention of our ICASSP2024 paper "G2G: Generalized Learning by Cross-Domain Knowledge Transfer for Federated Domain Generalization".

# Fedrated Domain Generalization
Code to reproduce the experiments of **G2G:Generalized Learning by Cross-Domain Knowledge Transfer for Federated Domain Generalization**.
## How to use it
* Clone or download the repository
### Install the requirement
 >  pip install -r requirements.txt
### Download VLCS, PACS and Office-Home datasets
* Download VLCS from https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md, extract, move it to datasets/VLCS/
* Download PACS from https://drive.google.com/uc?id=0B6x7gtvErXgfbF9CSk53UkRxVzg, move it to datasets/PACS/
* Download Office-Home from https://datasets.activeloop.ai/docs/ml/datasets/office-home-dataset/, move it to datasets/OfficeHome/
### Download Pre-trained models
* Download pre-trained AlexNet from https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth and move it to models/
* Download pre-trained Resnet50 from https://download.pytorch.org/models/resnet50-19c8e357.pth and move it to models/
* Download pre-trained VGG16 https://download.pytorch.org/models/vgg16-397923af.pth and move it to models/
### Running ours on VLCS
``` 
python main_warm.py --node_num 3  --device cuda:0 --dataset vlcs --classes 5 --lr 0.0008 --global_model Alexnet --local_model Alexnet --algorithm fed_mutual --R 50 --E 7 --pretrained True --batch_size 64 --iteration 0 
```
### Running ours on PACS
``` 
python main_warm.py --node_num 3  --device cuda:0 --dataset pacs --classes 7 --lr 0.0008 --global_model Alexnet --local_model Alexnet --algorithm fed_mutual --R 50 --E 7 --pretrained True --batch_size 32 --iteration 0 
```
### Running ours on Office-Home
``` 
python main_warm.py --node_num 3 --device cuda:0 --dataset office-home --classes 65 --lr 0.0008 --global_model ResNet50 --local_model ResNet50 --algorithm fed_mutual --R 50 --E 7 --batch_size 32 --iteration 0 
```

