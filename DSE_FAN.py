#-------------------------------------
#l2_norm:feature share在base的类间交换，训练时prototype进行augment，query样本不augment，以此来促使CNN中的C全连接自表达层学习来产生更加general的prototype样本进行query匹配
#-------------------------------------
import torch
import pprint
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
import os
import math
import argparse
import scipy.linalg as sl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import task_generator as tg
from utils import pprint, flip, perturb, GaussianNoise

# 参数设置
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
# 类别样本数基本设置
parser.add_argument("-c","--class_num",type = int, default = 5)#5
parser.add_argument("-s","--sample_num_per_class", type = int, default = 5)#5或者1
parser.add_argument("-b","--batch_num_per_class", type = int, default = 5)#XJTU有200个类(左右手看做不同类)，每个类总共10个样本，一般取15
parser.add_argument("-p","--save_path",type = str, default='XJTU_Conv4/models/DeNet')
# CNNEncoder主网络设置
parser.add_argument("-e","--episode",type = int, default = 5000)#4000
parser.add_argument("-v","--valid_episode", type = int, default = 100)#600
parser.add_argument("-t","--test_episode", type = int, default = 600)#600

parser.add_argument("-l","--learning_rate", type = float, default = 0.004)#0.004
parser.add_argument("-g","--gpu",type=int, default=1)
parser.add_argument("-f","--feature_dim",type = int, default = 64)#64
parser.add_argument("-pn","--p_norm", type=str, default=1)#1,2='fro','nuc'
parser.add_argument("-ga","--gamma", type=float, default=0.5)#0.5
parser.add_argument("-se","--seed", type=float, default=4)#seed 4
parser.add_argument("-la1","--lambd1", type=float, default=2000)#loss_dis,50
parser.add_argument("-la2","--lambd2", type=float, default=2)#2,loss_V
parser.add_argument("-la3","--lambd3", type=float, default=0.01)#0.01,loss_G
parser.add_argument("-la4","--lambd4", type=float, default=0.2)#0.2,loss_E
parser.add_argument("-la5","--lambd5", type=float, default=0.01)#0.01,loss_Recon
parser.add_argument("-gau","--gaussian", type=float, default=0.001)
# 参数及实验结果保存在字典
args = parser.parse_args()
pprint(vars(args))
trlog = {}
trlog['args'] = vars(args)#指令加入字典
trlog['train_loss_nets'] = []
trlog['max_valid_acc'] = 0.0
trlog['final_valid_acc'] = 0.0
trlog['reconstruct_error'] = 0.0

# 基本设置
CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, SAVE_PATH = args.class_num, args.sample_num_per_class, args.batch_num_per_class, args.save_path
# 主循环
EPISODE, VALID_EPISODE, TEST_EPISODE, LEARNING_RATE, GPU, FEATURE_DIM= args.episode, args.valid_episode, args.test_episode, args.learning_rate, args.gpu, args.feature_dim
SEED, P_NORM, GAMMA, LAMBD1, LAMBD2, LAMBD3, LAMBD4, LAMBD5, GAUSSIAN = args.seed, args.p_norm, args.gamma, args.lambd1, args.lambd2, args.lambd3, args.lambd4, args.lambd5, args.gaussian

# 函数
def construct_G(CLASS, NUM_BASE, NUM_QUERY):
    #构造图矩阵G
    if NUM_BASE ==1:
        NUM_BASE = 2
    subblock = [[1/NUM_BASE]*(NUM_BASE) for i in range(NUM_BASE)]
    append_subblock = [subblock for i in range(CLASS)]
    block = sl.block_diag(*append_subblock) #转为对角块，*表示输入需为列表解开的单独元素
    Block = torch.Tensor(block)
    W = torch.where(Block==0., torch.tensor(-1/((CLASS-1)*NUM_BASE)), Block) #1.默认float32
    d = sl.block_diag(*torch.sum(W,dim=0))
    D = torch.Tensor(d)
    G = D - W
    return G

class EnDecoder(nn.Module):
    def __init__(self, feature_dim, cls_num, sam_num):
        super(EnDecoder, self).__init__()
        self.Encodlayer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        # nn.ReLU(),
                        nn.MaxPool2d(2))
        self.Encodlayer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.ReLU(),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        # nn.ReLU(),
                        nn.MaxPool2d(2))
        self.Encodlayer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        # nn.ReLU()
                        )
        self.Encodlayer4 = nn.Sequential(
                        nn.Conv2d(64,feature_dim,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        # nn.ReLU()
                        )

        self.Decodlayer1 = nn.Sequential(
                        nn.ConvTranspose2d(feature_dim,64,kernel_size=3,padding=1,output_padding=0),
                        nn.ReLU(),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        # nn.ReLU()
                        )
        self.Decodlayer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,64,kernel_size=3,padding=1,output_padding=0),
                        nn.ReLU(),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        # nn.ReLU()
                        )
        self.Decodlayer3 = nn.Sequential(
                        nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=0,output_padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        # nn.ReLU()
                        )
        self.Decodlayer4 = nn.Sequential(
                        nn.ConvTranspose2d(64,64,kernel_size=3,stride=1,padding=1,output_padding=0),
                        nn.ReLU(),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        # nn.ReLU()
                        )
        self.Decodlayer5 = nn.Sequential(
                        nn.ConvTranspose2d(64,3,kernel_size=3,stride=2,padding=1,output_padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(3, momentum=1, affine=True),
                        # nn.ReLU()
                        )   

        self.cls_num, self.sam_num = cls_num, sam_num
        if self.sam_num ==1:
            self.sam_num = 2
        self.linear = nn.Linear(in_features=self.cls_num*self.sam_num, out_features=self.cls_num*self.sam_num, bias=False)

        # torch.manual_seed(1)
        # w = torch.randn(CLASS_NUM*SAMPLE_NUM_PER_CLASS, CLASS_NUM*SAMPLE_NUM_PER_CLASS)
        # self.W = nn.Parameter(Variable(torch.Tensor(w), requires_grad=True))

    def forward(self,x):
        e = self.Encodlayer1(x)
        e = self.Encodlayer2(e)
        e = self.Encodlayer3(e)
        e = self.Encodlayer4(e)

        emb = e.view(e.size(0),-1) # 25*(FEATURE_DIM*19*19)
        W = self.linear.weight - torch.diag(torch.diag(self.linear.weight))
        emb_p = torch.matmul(W, emb)
        emb_v = emb - emb_p

        r_e = self.Decodlayer1((emb_p+emb_v).view(e.size()))
        r_e = self.Decodlayer2(r_e)
        r_e = self.Decodlayer3(r_e)
        r_e = self.Decodlayer4(r_e)
        r_x = self.Decodlayer5(r_e)
        return emb, emb_p, emb_v, self.linear.weight, r_x-x

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
        # m.weight.data.uniform_(0, 1)
        tg.setup_seed(SEED)
        m.weight.data.normal_(0, 0.01)
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias == True:
            m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("Begining " + str(CLASS_NUM) + " Way " + str(SAMPLE_NUM_PER_CLASS) + " Shot Training")
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metavalid_folders,metatest_folders = tg.mini_imagenet_folders() #200:60:140,按照3:1:2取整划分

    # Step 2: init neural networks
    print("initing neural networks")
    tg.setup_seed(SEED)

    feature_encoder = EnDecoder(FEATURE_DIM, CLASS_NUM, SAMPLE_NUM_PER_CLASS)
    feature_encoder.apply(weights_init)
    feature_encoder.cuda(GPU)
    graph_G = construct_G(CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS).cuda(GPU)
    makenoise = GaussianNoise(CLASS_NUM*SAMPLE_NUM_PER_CLASS, (3, 84, 84), std=0.05)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=600,gamma=GAMMA)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.wd)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=self.wd)
    # Step 3: build graph
    print("Begain Training...")
    last_accuracy = 0.0
    last_reconstruct = 1000000
    for episode in range(EPISODE):
        # 获取样本sampling
        train_task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(train_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(train_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)      
        samples, sample_labels = sample_dataloader.__iter__().next() #25*3*84*84
        batches, batch_labels = batch_dataloader.__iter__().next()
        # 特征分解decomposition
        if SAMPLE_NUM_PER_CLASS == 1:
            samples = torch.cat((samples, makenoise(samples, GAUSSIAN)), dim=1).view(CLASS_NUM*SAMPLE_NUM_PER_CLASS*2,3,84,84)
            batches = torch.cat((batches, makenoise(batches, GAUSSIAN)), dim=0)
        sample_features, P_sample, V_sample, W, R_errors = feature_encoder(Variable(samples).cuda(GPU)) # 25*(64*19*19)二维
        batch_features, _, _, _, _ = feature_encoder(Variable(batches).cuda(GPU)) # 25*(64*19*19)二维
        prototype_classes = tg.Feature_transfer(sample_features, P_sample, V_sample, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS) #base和query的P和V互相交叉得到衍生样本特征
        # training main networks
        distance = tg.euclidean_dist(batch_features[:CLASS_NUM*BATCH_NUM_PER_CLASS], prototype_classes, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        log_distance = -F.log_softmax(-distance, dim=1)
        index_labels = batch_labels.view(BATCH_NUM_PER_CLASS*CLASS_NUM,-1).cuda(GPU)
        loss_dis = log_distance.gather(1, index_labels).squeeze().view(-1).mean()
        # (3) 总体损失函数优化
        loss_W, loss_V,loss_G,loss_E,loss_R=torch.abs(W).sum(),torch.pow(V_sample,2).sum(),torch.trace((P_sample.t().mm(graph_G)).mm(P_sample)),torch.pow(P_sample,2).sum(),torch.pow(R_errors,2).sum()#torch.diag((P_sample.t().mm(graph_G)).mm(P_sample)).sum()
        # loss_W, loss_V,loss_G,loss_E,loss_R=torch.norm(W,p=P_NORM),torch.norm(V_sample,p=2),torch.trace((P_sample.t().mm(graph_G)).mm(P_sample)),torch.norm(P_sample,p=2),torch.norm(R_errors,p=2)
        total_loss = (loss_W + LAMBD2*loss_V) + LAMBD1*loss_dis + (LAMBD3*loss_G + LAMBD4*loss_E) + LAMBD5*loss_R
        feature_encoder_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)
        feature_encoder_optim.step()
        feature_encoder_scheduler.step()
        # (4) 损失函数数值显示
        if (episode+1)%50 == 0:
            print("episode", episode+1, "in", EPISODE, ": Total_loss", total_loss.item(), ", W", loss_W.item(), ", V", loss_V.item(), ", Dis", loss_dis.item(), ", G", loss_G.item(), ", E", loss_E.item(), ", R", loss_R.item())
        # 验证结果显示validation
        if (episode+1)%200 == 0: #5000
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
                    if SAMPLE_NUM_PER_CLASS == 1:
                        base = torch.cat((base, makenoise(base, GAUSSIAN)), dim=1).view(CLASS_NUM*SAMPLE_NUM_PER_CLASS*2,3,84,84)
                        query = torch.cat((query, makenoise(query, GAUSSIAN)), dim=0)
                    # 特征分解decomposition
                    base_features, P_base, V_base, _, _ = feature_encoder(Variable(base).cuda(GPU)) # 25*(64*19*19)二维
                    query_features, _, _, _ , _ = feature_encoder(Variable(query).cuda(GPU)) # 25*(64*19*19)二维
                    prototype_valid_classes = tg.Feature_transfer(base_features, P_base, V_base, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS) #base和query的P和V互相交叉得到衍生样本特征
                    # training main networks
                    valid_distance = tg.euclidean_dist(query_features[:CLASS_NUM*BATCH_NUM_PER_CLASS], prototype_valid_classes, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
                    log_valid_dis = -F.log_softmax(-valid_distance, dim=1)
                    _,predict_labels = torch.min(log_valid_dis,1)
                    accuracy = torch.eq(predict_labels, query_labels.cuda(GPU)).float().mean()
                    accuracies.append(accuracy.item()) # item转化为数字,numpy()转化为np数字

                valid_accuracy, h = tg.mean_confidence_interval(accuracies) #非张量数值
                print("valid accuracy:", valid_accuracy,", h:", h)
                if valid_accuracy >= last_accuracy:
                    # 保存当前最佳网络
                    if not os.path.exists(SAVE_PATH):
                        os.makedirs(SAVE_PATH)
                    torch.save(feature_encoder.state_dict(),str(SAVE_PATH+"/feature_encoder_"+str(CLASS_NUM)+"way_"+str(SAMPLE_NUM_PER_CLASS)+"shot_"+"max-valid.pkl"))
                    last_accuracy = valid_accuracy

                    # 保存训练和验证结果
                    trlog['max_valid_acc'] = {'valid_accuracy':valid_accuracy, 'h':h}
                    torch.save(trlog, os.path.join(SAVE_PATH, str(str(CLASS_NUM)+"way_"+str(SAMPLE_NUM_PER_CLASS)+"shot_"+'maxvalid_trlog')))
                    print("saved networks for max_valid episode:",episode+1,"accuracy:",valid_accuracy)

                if loss_R <= last_reconstruct:
                    # 保存当前最佳网络
                    torch.save(feature_encoder.state_dict(),str(SAVE_PATH+"/feature_encoder_"+str(CLASS_NUM)+"way_"+str(SAMPLE_NUM_PER_CLASS)+"shot_"+"min_reconstruct_error.pkl"))
                    last_reconstruct = loss_R

                    # 保存训练和验证结果
                    trlog['reconstruct_error'] = {loss_R}
                    torch.save(trlog, os.path.join(SAVE_PATH, str(str(CLASS_NUM)+"way_"+str(SAMPLE_NUM_PER_CLASS)+"shot_"+'min_reconstruct_error_trlog')))
                    print("saved networks for min_reconstruct_error episode:",episode+1,",error:",loss_R.item())
                
                
        if episode+1 == EPISODE:
            torch.save(feature_encoder.state_dict(),str(SAVE_PATH+"/feature_encoder_"+str(CLASS_NUM)+"way_"+str(SAMPLE_NUM_PER_CLASS)+"shot_"+"final-episode.pkl"))
            # 保存训练和验证结果
            trlog['train_loss_nets'].append(total_loss)
            trlog['final_valid_acc'] = {'valid_accuracy':valid_accuracy, 'h':h}
            torch.save(trlog, os.path.join(SAVE_PATH, str(str(CLASS_NUM)+"way_"+str(SAMPLE_NUM_PER_CLASS)+"shot_"+'finalvalid_trlog')))
            print("saved networks and args for final_valid episode:",episode+1)
        
    
    # 测试版本1
    with torch.no_grad():
        print("Begining " + str(CLASS_NUM) +" Way " + str(SAMPLE_NUM_PER_CLASS) + " Shot Max-Valid Model Test")
        _,_,metatest_folders = tg.mini_imagenet_folders()
        feature_encoder_test1 = EnDecoder(FEATURE_DIM, CLASS_NUM, SAMPLE_NUM_PER_CLASS)
        feature_encoder_test1.cuda(GPU)

        if os.path.exists(str(SAVE_PATH + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot_"+"max-valid.pkl")):
            feature_encoder_test1.load_state_dict(torch.load(str(SAVE_PATH + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot_"+"max-valid.pkl")))
            print("load max-valid network successfully")
        else:
            print("none existing network")

        accuracies = []
        for i in range(TEST_EPISODE):
            # 获取样本sampling
            test_task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
            base_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
            query_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
            base, base_labels = base_dataloader.__iter__().next()
            query, query_labels = query_dataloader.__iter__().next()
            if SAMPLE_NUM_PER_CLASS == 1:
                base = torch.cat((base, makenoise(base, GAUSSIAN)), dim=1).view(CLASS_NUM*SAMPLE_NUM_PER_CLASS*2,3,84,84)
                query = torch.cat((query, makenoise(query, GAUSSIAN)), dim=0)
            # 特征分解decomposition
            base_features, P_base, V_base, _, _ = feature_encoder_test1(Variable(base).cuda(GPU)) # 25*(64*19*19)二维
            query_features, _, _, _, _ = feature_encoder_test1(Variable(query).cuda(GPU)) # 25*(64*19*19)二维
            prototype_test_classes = tg.Feature_transfer(base_features, P_base, V_base, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS) #base和query的P和V互相交叉得到衍生样本特征
            # training main networks
            test_distance = tg.euclidean_dist(query_features[:CLASS_NUM*BATCH_NUM_PER_CLASS], prototype_test_classes, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
            log_test_dis = -F.log_softmax(-test_distance, dim=1)
            _,predict_labels = torch.min(log_test_dis,1)
            accuracy = torch.eq(predict_labels, query_labels.cuda(GPU)).float().mean()
            accuracies.append(accuracy.item())

        test_accuracy, h = tg.mean_confidence_interval(accuracies) #非张量数值
        print("test accuracy:", test_accuracy,", h:", h)
        
        # 测试版本2
        print("Begining " + str(CLASS_NUM) +" Way " + str(SAMPLE_NUM_PER_CLASS) + " Shot Final-Valid Model Test")
        _,_,metatest_folders = tg.mini_imagenet_folders()
        feature_encoder_test2 = EnDecoder(FEATURE_DIM, CLASS_NUM, SAMPLE_NUM_PER_CLASS)
        feature_encoder_test2.cuda(GPU)

        if os.path.exists(str(SAVE_PATH + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS)+"shot_"+"final-episode.pkl")):
            feature_encoder_test2.load_state_dict(torch.load(str(SAVE_PATH + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot_"+"final-episode.pkl")))
            print("load final-valid network successfully")
        else:
            print("none existing network")

        accuracies = []
        for i in range(TEST_EPISODE):
            # 获取样本sampling
            test_task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
            base_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
            query_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
            base, base_labels = base_dataloader.__iter__().next()
            query, query_labels = query_dataloader.__iter__().next()
            if SAMPLE_NUM_PER_CLASS == 1:
                base = torch.cat((base, makenoise(base, GAUSSIAN)), dim=1).view(CLASS_NUM*SAMPLE_NUM_PER_CLASS*2,3,84,84)
                query = torch.cat((query, makenoise(query, GAUSSIAN)), dim=0)
            # 特征分解decomposition
            base_features, P_base, V_base, _, _ = feature_encoder_test2(Variable(base).cuda(GPU)) # 25*(64*19*19)二维
            query_features, _, _, _, _ = feature_encoder_test2(Variable(query).cuda(GPU)) # 25*(64*19*19)二维
            prototype_test_classes = tg.Feature_transfer(base_features, P_base, V_base, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS) #base和query的P和V互相交叉得到衍生样本特征
            # training main networks
            test_distance = tg.euclidean_dist(query_features[:CLASS_NUM*BATCH_NUM_PER_CLASS], prototype_test_classes, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
            log_test_dis = -F.log_softmax(-test_distance, dim=1)
            _,predict_labels = torch.min(log_test_dis,1)
            accuracy = torch.eq(predict_labels, query_labels.cuda(GPU)).float().mean()
            accuracies.append(accuracy.item())

        test_accuracy, h = tg.mean_confidence_interval(accuracies) #非张量数值
        print("test accuracy:", test_accuracy,", h:", h)


if __name__ == '__main__':
    main()