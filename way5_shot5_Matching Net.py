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
from torch.nn.utils import clip_grad_norm_
# 参数设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
# 类别样本数基本设置
parser.add_argument("-w","--class_num",type = int, default = 5)#5
parser.add_argument("-s","--sample_num_per_class", type = int, default = 5)#5或者1
parser.add_argument("-b","--batch_num_per_class", type = int, default = 5)#XJTU有200个类(左右手看做不同类)，每个类总共10个样本，一般取15
parser.add_argument("-p","--save_path",type = str, default='XJTU_Conv4/models/MatchNet')
# CNNEncoder主网络设置
parser.add_argument("-e","--episode",type = int, default =4000)#500000
parser.add_argument("-v","--valid_episode", type = int, default = 100)#600
parser.add_argument("-t","--test_episode", type = int, default = 600)#600
parser.add_argument("-l","--learning_rate", type = float, default = 0.0002)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-f","--feature_dim",type = int, default = 64)#可适当改大
parser.add_argument('--fce', type=bool, default=True)
parser.add_argument('--lstm-layers', default=1, type=int)
parser.add_argument('--num_input_channels', default=3, type=int)
parser.add_argument('--lstm_input_size', default=1600, type=int)
parser.add_argument('--unrolling_steps', default=3, type=int)
parser.add_argument('--distance', default='l2')
parser.add_argument('--epsilon', type = float, default = 1e-8)
# 参数及实验结果保存在字典
args = parser.parse_args()
trlog = {}# pprint(vars(args))
trlog['args'] = vars(args)#指令加入字典
trlog['train_loss_nets'] = []
trlog['max_valid_acc'] = 0.0

# 基本设置
CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS= args.class_num, args.sample_num_per_class, args.batch_num_per_class
# 主循环
EPISODE, VALID_EPISODE, TEST_EPISODE, LEARNING_RATE, GPU = args.episode, args.valid_episode, args.test_episode, args.learning_rate, args.gpu

# 函数
class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def get_few_shot_encoder(num_input_channels=1,dim=64) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks

    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, dim),
        Flatten(),
    )

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

class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda()#.double()
        c = torch.zeros(batch_size, embedding_dim).cuda()#.double()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries
            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)
            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)
            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))
        h = h_hat + queries
        return h

class MatchingNetwork(nn.Module):
    def __init__(self, n: int, k: int, q: int, fce: bool, num_input_channels: int,
                 lstm_layers: int, lstm_input_size: int, unrolling_steps: int, feature_dim: int, device: torch.device):
        """Creates a Matching Network as described in Vinyals et al.
        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            fce: Whether or not to us fully conditional embeddings
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and Attention LSTM. This is determined by the embedding
                dimension of the few shot encoder which is in turn determined by the size of the input data. Hence we
                have Omniglot -> 64, miniImageNet -> 1600.
            unrolling_steps: Number of unrolling steps to run the Attention LSTM
            device: Device on which to run computation
        """
        super(MatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.fce = fce
        self.num_input_channels = num_input_channels
        self.encoder = get_few_shot_encoder(self.num_input_channels, feature_dim).to(device)
        if self.fce:
            self.g = BidrectionalLSTM(lstm_input_size, lstm_layers).to(device)
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps=unrolling_steps).to(device)

    def forward(self, x):
        pass

def main():
    # Step 1: init data folders
    print("Begining " + str(CLASS_NUM) +" Way " + str(SAMPLE_NUM_PER_CLASS) + " Shot Training on XJTU Database")
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metavalid_folders,metatest_folders = tg.mini_imagenet_folders()#200:60:140,按照3:1:2取整划分

    # Step 2: init neural networks
    print("init neural networks")
    model = MatchingNetwork(SAMPLE_NUM_PER_CLASS, CLASS_NUM, BATCH_NUM_PER_CLASS, args.fce, args.num_input_channels,
                        lstm_layers=args.lstm_layers,lstm_input_size=args.lstm_input_size,
                        unrolling_steps=args.unrolling_steps,feature_dim=args.feature_dim,device=GPU)
    # model.apply(weights_init)

    feature_encoder_optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=1000,gamma=0.5)
    loss_fn = torch.nn.NLLLoss().cuda()

    # Step 3: build graph
    print("Begain Training...")
    last_accuracy = 0.0
    for episode in range(EPISODE):
        # 获取样本sampling
        model.train()

        train_task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(train_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(train_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=False)
        
        samples, sample_labels = sample_dataloader.__iter__().next() #25*3*84*84
        batches, batch_labels = batch_dataloader.__iter__().next()

        # 特征分解decomposition
        support = model.encoder(samples.cuda(GPU))
        queries = model.encoder(batches.cuda(GPU))
        # Optionally apply full context embeddings
        if args.fce:
            # LSTM requires input of shape (seq_len, batch, input_size). `support` is of
            # shape (k_way * n_shot, embedding_dim) and we want the LSTM to treat the
            # support set as a sequence so add a single dimension to transform support set
            # to the shape (k_way * n_shot, 1, embedding_dim) and then remove the batch dimension
            # afterwards

            # Calculate the fully conditional embedding, g, for support set samples as described
            # in appendix A.2 of the paper. g takes the form of a bidirectional LSTM with a
            # skip connection from inputs to outputs
            support, _, _ = model.g(support.unsqueeze(1))
            support = support.squeeze(1)

            # Calculate the fully conditional embedding, f, for the query set samples as described in appendix A.1 of the paper.
            queries = model.f(support, queries)

        # Efficiently calculate distance between all queries and all prototypes
        distances = tg.pairwise_distances(queries, support, args.distance, args.epsilon)

        # Calculate "attention" as softmax over support-query distances
        # attention = (-distances).softmax(dim=1)
        # y_pred = tg.matching_net_predictions(attention, sample_labels, SAMPLE_NUM_PER_CLASS,CLASS_NUM,BATCH_NUM_PER_CLASS)
        # clipped_y_pred = y_pred.clamp(args.epsilon, 1 - args.epsilon)
        # train_loss = loss_fn(clipped_y_pred.log(), batch_labels.cuda(GPU))

        y_pred = tg.matching_net_predictions(distances, sample_labels, SAMPLE_NUM_PER_CLASS,CLASS_NUM,BATCH_NUM_PER_CLASS)
        log_distance = -F.log_softmax(-y_pred, dim=1)
        index_labels = batch_labels.view(BATCH_NUM_PER_CLASS*CLASS_NUM,-1).cuda(GPU)
        train_loss = log_distance.gather(1, index_labels).squeeze().view(-1).mean()

        # I found training to be quite unstable so I clip the norm of the gradient to be at most 1
        feature_encoder_optim.zero_grad()
        train_loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        # Take gradient step
        feature_encoder_optim.step()
        feature_encoder_scheduler.step()

        # model.eval()
        # trian_acc = torch.eq(y_pred.argmax(dim=-1), batch_labels.cuda(GPU)).sum()/ y_pred.shape[0]
        
        # (4) 损失函数数值显示
        if (episode+1)%50 == 0:
            # print('Episode{:4d} in EPISODE {:4d}: loss {:4f}, acc {:4f}'.format(episode+1, EPISODE, loss.item(), acc.item()))
            # print(f'Episode{episode+1} in EPISODE {EPISODE}: trian_loss {train_loss.item()}, trian_acc {trian_acc.item()}')
            print(f'Episode {episode+1} in EPISODE {EPISODE}: train_loss {train_loss.item()}')

        # 验证结果显示validation
        if (episode+1)%500 == 0:#5000
            with torch.no_grad():
                print("Validating...")
                valid_acc = []
                for i in range(VALID_EPISODE):
                    # 获取样本sampling
                    valid_task = tg.MiniImagenetTask(metavalid_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
                    base_dataloader = tg.get_mini_imagenet_data_loader(valid_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                    query_dataloader = tg.get_mini_imagenet_data_loader(valid_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=False)
                    base, base_labels = base_dataloader.__iter__().next()
                    query, query_labels = query_dataloader.__iter__().next()  
                    support = model.encoder(base.cuda(GPU))
                    queries = model.encoder(query.cuda(GPU))
                    if args.fce:
                        support, _, _ = model.g(support.unsqueeze(1))
                        support = support.squeeze(1)
                        queries = model.f(support, queries)
                    distances = tg.pairwise_distances(queries, support, args.distance, args.epsilon)
                    attention = (-distances).softmax(dim=1)
                    y_pred = tg.matching_net_predictions(attention, base_labels, SAMPLE_NUM_PER_CLASS,CLASS_NUM,BATCH_NUM_PER_CLASS)

                    valid_acc_temp = torch.eq(y_pred.argmax(dim=-1), query_labels.cuda(GPU)).sum()/y_pred.shape[0]
                    valid_acc.append(valid_acc_temp.item())

                valid_acc_aver, valid_acc_h = tg.mean_confidence_interval(valid_acc) #非张量数值
                print(f'Validation at episode {episode+1}: valid_acc_aver {valid_acc_aver.item()}, valid_acc_h {valid_acc_h}')

                if valid_acc_aver > last_accuracy:
                    # 保存当前最佳网络
                    save_path = args.save_path
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(model.state_dict(),str(args.save_path + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    last_accuracy = valid_acc_aver

                    # 保存训练和验证结果
                    trlog['train_loss_nets'].append(train_loss)
                    trlog['max_valid_acc'] = {'valid_accuracy':valid_acc_aver, 'h':valid_acc_h}
                    torch.save(trlog, os.path.join(args.save_path, 'trlog'))
                    print("saved networks and args for episode:",episode+1)

    # 开始测试 1
    # Step 1: init data folders
    print("Begining " + str(CLASS_NUM) +" Way " + str(SAMPLE_NUM_PER_CLASS) + " Shot Last-Valid Acc-Model Test")

    test_acc = []
    print("Testing 1...")
    for i in range(TEST_EPISODE):
        # 获取样本sampling
        test_task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        base_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        query_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
        base, base_labels = base_dataloader.__iter__().next()
        query, query_labels = query_dataloader.__iter__().next()
        # 特征分解decomposition
        support = model.encoder(base.cuda(GPU))
        queries = model.encoder(query.cuda(GPU))
        if args.fce:
            support, _, _ = model.g(support.unsqueeze(1))
            support = support.squeeze(1)
            queries = model.f(support, queries)
        distances = tg.pairwise_distances(queries, support, args.distance, args.epsilon)
        attention = (-distances).softmax(dim=1)
        y_pred = tg.matching_net_predictions(attention, base_labels, SAMPLE_NUM_PER_CLASS,CLASS_NUM,BATCH_NUM_PER_CLASS)

        test_acc_temp = torch.eq(y_pred.argmax(dim=-1), query_labels.cuda(GPU)).sum()/y_pred.shape[0]
        test_acc.append(test_acc_temp.item())

    test_acc_aver, test_acc_h = tg.mean_confidence_interval(test_acc) #非张量数值
    print(f'Test at episode {episode+1}: test_acc_aver {test_acc_aver.item()}, test_acc_h {test_acc_h}')

    # 开始测试 2
    # Step 1: init data folders
    print("Begining " + str(CLASS_NUM) +" Way " + str(SAMPLE_NUM_PER_CLASS) + " Shot Best-Valid Acc-Model Test")
    model = MatchingNetwork(SAMPLE_NUM_PER_CLASS, CLASS_NUM, BATCH_NUM_PER_CLASS, args.fce, args.num_input_channels,
                        lstm_layers=args.lstm_layers,lstm_input_size=args.lstm_input_size,
                        unrolling_steps=args.unrolling_steps,feature_dim=args.feature_dim,device=GPU)
    if os.path.exists(str(args.save_path + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        model.load_state_dict(torch.load(str(args.save_path + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load network successfully")
    else:
        print("none existing network")

    test_acc = []
    print("Testing 2...")
    for i in range(TEST_EPISODE):
        # 获取样本sampling
        test_task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        base_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        query_dataloader = tg.get_mini_imagenet_data_loader(test_task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
        base, base_labels = base_dataloader.__iter__().next()
        query, query_labels = query_dataloader.__iter__().next()
        # 特征分解decomposition
        support = model.encoder(base.cuda(GPU))
        queries = model.encoder(query.cuda(GPU))
        if args.fce:
            support, _, _ = model.g(support.unsqueeze(1))
            support = support.squeeze(1)
            queries = model.f(support, queries)
        distances = tg.pairwise_distances(queries, support, args.distance, args.epsilon)
        attention = (-distances).softmax(dim=1)
        y_pred = tg.matching_net_predictions(attention, base_labels, SAMPLE_NUM_PER_CLASS,CLASS_NUM,BATCH_NUM_PER_CLASS)

        test_acc_temp = torch.eq(y_pred.argmax(dim=-1), query_labels.cuda(GPU)).sum()/y_pred.shape[0]
        test_acc.append(test_acc_temp.item())

    test_acc_aver, test_acc_h = tg.mean_confidence_interval(test_acc) #非张量数值
    print(f'Test at episode {episode+1}: test_acc_aver {test_acc_aver.item()}, test_acc_h {test_acc_h}')

if __name__ == '__main__':
    main()