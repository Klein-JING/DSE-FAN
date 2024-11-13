#-------------------------------------
# Decomposing Networks with feature alignment and transfer for few-shot palmprint learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------
import torch
import pprint
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import scipy.linalg as sl
import matplotlib.pyplot as plt
import task_generator_test as tg

# 参数设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
# 类别样本数基本设置
parser.add_argument("-w","--class_num",type = int, default = 5)#5
parser.add_argument("-s","--sample_num_per_class", type = int, default = 1)#5或者1
parser.add_argument("-b","--batch_num_per_class", type = int, default = 1)#XJTU有200个类(左右手看做不同类)，每个类总共10个样本，一般取15
parser.add_argument("-p","--save_path",type = str, default='XJTU_Conv4/models/ProtoNet')
# CNNEncoder主网络设置
parser.add_argument("-e","--episode",type = int, default =2000)#500000
parser.add_argument("-v","--valid_episode", type = int, default = 100)#600
parser.add_argument("-t","--test_episode", type = int, default = 600)#600
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-f","--feature_dim",type = int, default = 64)#可适当改大
parser.add_argument("-a","--lambd", type=float, default=1)
# 参数及实验结果保存在字典
args = parser.parse_args()
trlog = {}# pprint(vars(args))
trlog['args'] = vars(args)#指令加入字典
trlog['train_loss_nets'] = []
trlog['max_valid_acc'] = 0.0

# 基本设置
CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS= args.class_num, args.sample_num_per_class, args.batch_num_per_class
# 主循环
EPISODE, VALID_EPISODE, TEST_EPISODE, LEARNING_RATE, GPU, FEATURE_DIM, LAMBD= args.episode, args.valid_episode, args.test_episode, args.learning_rate, args.gpu, args.feature_dim, args.lambd

# 函数
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self, feature_dim):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,feature_dim,kernel_size=3,padding=1),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),-1) # 25*(FEATURE_DIM*19*19)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    # Step 1: init data folders
    print("Begining " + str(CLASS_NUM) +" Way " + str(SAMPLE_NUM_PER_CLASS) + " Shot Training on XJTU Database")
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metavalid_folders,_ = tg.mini_imagenet_folders()#200:60:140,按照3:1:2取整划分

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder(FEATURE_DIM)
    feature_encoder.apply(weights_init)
    feature_encoder.cuda(GPU)
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=1000,gamma=0.5)

    # Step 3: build graph
    print("Begain Training...")
    last_accuracy = 0.0
    for episode in range(EPISODE):
        # 获取样本sampling
        train_task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(train_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(train_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
        
        samples, sample_labels = sample_dataloader.__iter__().next() #25*3*84*84
        batches, batch_labels = batch_dataloader.__iter__().next()

        # 特征分解decomposition
        sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 25*64*19*19
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,-1)
        sample_features = torch.sum(sample_features,1).squeeze(1)/SAMPLE_NUM_PER_CLASS
        batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5
        
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1).transpose(0,1)
        global_distance = torch.sum((sample_features_ext - batch_features_ext)**2,2) #变成2维：BATCH_NUM_PER_CLASS x CLASS_NUM
        # global_distance = torch.pow(sample_features_ext - batch_features_ext, 2).sum(2)
        log_global_distance = -F.log_softmax(-global_distance, dim=1)

        # one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).long().cuda(GPU))
        # global_loss = (log_global_distance*one_hot_labels).mean()
        global_loss = log_global_distance.gather(1, batch_labels.view(BATCH_NUM_PER_CLASS*CLASS_NUM,-1).cuda(GPU)).squeeze().mean()
        total_loss = LAMBD*global_loss
        # (3) 总体损失函数优化
        feature_encoder_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)
        feature_encoder_optim.step()
        feature_encoder_scheduler.step()
        # (4) 损失函数数值显示
        if (episode+1)%50 == 0:
            print("episode", episode+1, "in", EPISODE,": total_loss", total_loss.item())

        # 验证结果显示validation
        if (episode+1)%500 == 0:#5000
            with torch.no_grad():
                print("Validating...")
                accuracies = []
                for i in range(VALID_EPISODE):
                    # 获取样本sampling
                    valid_task = tg.MiniImagenetTask(metavalid_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
                    base_dataloader = tg.get_mini_imagenet_data_loader(valid_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                    query_dataloader = tg.get_mini_imagenet_data_loader(valid_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
                    base, base_labels = base_dataloader.__iter__().next()
                    query, query_labels = query_dataloader.__iter__().next()
                    # 特征分解decomposition
                    base_features = feature_encoder(Variable(base).cuda(GPU)) # 25*(64*19*19)二维
                    query_features = feature_encoder(Variable(query).cuda(GPU)) # 25*(64*19*19)二维
                    base_features_3d = base_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,-1)
                    base_features_prototype = torch.sum(base_features_3d, 1)/SAMPLE_NUM_PER_CLASS#变成二维
                    base_features_prototype_extend = base_features_prototype.unsqueeze(0).repeat(CLASS_NUM*BATCH_NUM_PER_CLASS,1,1) #变成3维:(BATCH_NUM_PER_CLASS*CLASS_NUM) x CLASS x Dimenstion       
                    query_features_meta_extend = query_features.unsqueeze(0).repeat(CLASS_NUM,1,1).transpose(0,1)
                    global_dis = torch.sum((query_features_meta_extend - base_features_prototype_extend)**2,2) #变成2维：BATCH_NUM_PER_CLASS x CLASS_NUM
                    log_global_dis = -F.log_softmax(-global_dis, dim=1)
                    # (3) 总体损失函数求最大标签      
                    _,predict_labels = torch.min(log_global_dis,1)
                    accuracy = torch.sum(predict_labels == query_labels.cuda())/CLASS_NUM/BATCH_NUM_PER_CLASS
                    accuracies.append(accuracy.item()) # item转化为数字,numpy()转化为np数字

                valid_accuracy, h = mean_confidence_interval(accuracies) #非张量数值
                print("valid accuracy:", valid_accuracy,", h:", h)

                if valid_accuracy > last_accuracy:
                    # 保存当前最佳网络
                    save_path = args.save_path
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(feature_encoder.state_dict(),str(args.save_path + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    last_accuracy = valid_accuracy

                    # 保存训练和验证结果
                    trlog['train_loss_nets'].append(total_loss)
                    trlog['max_valid_acc'] = {'valid_accuracy':last_accuracy, 'h':h}
                    torch.save(trlog, os.path.join(args.save_path, 'trlog'))
                    print("saved networks and args for episode:",episode+1)

    # 开始测试
    # Step 1: init data folders
    print("Begining " + str(CLASS_NUM) +" Way " + str(SAMPLE_NUM_PER_CLASS) + " Shot Best-ValidAcc-Model Test")
    # init character folders for dataset construction
    _,_,metatest_folders = tg.mini_imagenet_folders()
    feature_encoder = CNNEncoder(FEATURE_DIM)
    feature_encoder.apply(weights_init)
    feature_encoder.cuda(GPU)


    if os.path.exists(str(args.save_path + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str(args.save_path + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load network successfully")
    else:
        print("none existing network")

    accuracies = []
    print("Testing...")
    for i in range(TEST_EPISODE):
        # 获取样本sampling
        test_task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        base_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        query_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
        base, base_labels = base_dataloader.__iter__().next()
        query, query_labels = query_dataloader.__iter__().next()
        # 特征分解decomposition
        base_features = feature_encoder(Variable(base).cuda(GPU)) # 25*(64*19*19)二维
        query_features = feature_encoder(Variable(query).cuda(GPU)) # 25*(64*19*19)二维
        base_features_3d = base_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,-1)
        base_features_prototype = torch.sum(base_features_3d, 1)/SAMPLE_NUM_PER_CLASS#变成二维
        base_features_prototype_extend = base_features_prototype.unsqueeze(0).repeat(CLASS_NUM*BATCH_NUM_PER_CLASS,1,1) #变成3维:(BATCH_NUM_PER_CLASS*CLASS_NUM) x CLASS x Dimenstion       
        query_features_meta_extend = query_features.unsqueeze(0).repeat(CLASS_NUM,1,1).transpose(0,1)
        global_dis = torch.sum((query_features_meta_extend - base_features_prototype_extend)**2,2) #变成2维：BATCH_NUM_PER_CLASS x CLASS_NUM
        log_global_dis = -F.log_softmax(-global_dis, dim=1)
        # (3) 总体损失函数求最大标签      
        _,predict_labels = torch.min(log_global_dis,1)
        accuracy = torch.sum(predict_labels == query_labels.cuda())/CLASS_NUM/BATCH_NUM_PER_CLASS
        accuracies.append(accuracy.item()) # item转化为数字,numpy()转化为np数字

    test_accuracy, h = mean_confidence_interval(accuracies) #非张量数值
    print("test accuracy:", test_accuracy,", h:", h)

if __name__ == '__main__':
    main()