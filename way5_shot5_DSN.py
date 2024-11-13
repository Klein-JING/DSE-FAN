import argparse
import os.path as osp
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import pprint, set_gpu, Averager, count_acc, flip

import numpy as np
import scipy as sp
import scipy.stats

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MiniImageNet(Dataset):
    def __init__(self, setname):
        data = []
        label = []
        lb = -1
        data_root = os.getcwd()
        ROOT_PATH = os.path.abspath(os.path.join(data_root, "../../datas/XJTU")) 
        classes = osp.join(ROOT_PATH, setname)
        for cls in os.listdir(classes):
            lb += 1
            for cls_individual in os.listdir(os.path.join(classes, cls)):
                img_path = os.path.join(classes, cls, cls_individual)
                data.append(img_path)
                label.append(lb)
        self.data = data
        self.label = label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        img  =Image.open(path).convert('RGB')
        img = img.resize((84, 84)).convert('RGB')
        image = self.transform(img)
        return image, label

class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            if len(ind) > 4:
                self.m_ind.append(ind)
    def __len__(self):
        return self.n_batch  
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)#stack需输入元素为tensor列表,或元组,t()表示转置,reshape后每类依次各取一个元素排列如tensor([1, 2, 1, 2])
            #for i in range(1000):
            yield batch#生成一个迭代序列，每个batch由n_cls*n_per个元素组成

class Subspace_Projection(nn.Module):
    def __init__(self, num_dim=5):
        super().__init__()
        self.num_dim = num_dim
    def create_subspace(self, supportset_features, class_size, sample_size):
        all_hyper_planes = []
        means = []
        for ii in range(class_size):
            num_sample = sample_size
            all_support_within_class_t = supportset_features[ii]#每一类的所有样本组成
            meann = torch.mean(all_support_within_class_t, dim=0)#按第一维度取平均值
            means.append(meann)
            all_support_within_class_t = all_support_within_class_t - meann.unsqueeze(0).repeat(num_sample, 1)#[1,2]->[[1,2]]->[[1,2],[1,2]],行样本减所有样本平均[[1类1样本:1xM],[1类2样本:1xM]](2维)
            all_support_within_class = torch.transpose(all_support_within_class_t, 0, 1)#相当于转置,行数为特征维度
            uu, s, v = torch.svd(all_support_within_class.double(), some=False)
            uu = uu.float()
            all_hyper_planes.append(uu[:, :self.num_dim])
        all_hyper_planes = torch.stack(all_hyper_planes, dim=0)#堆叠起来，0维为类数
        means = torch.stack(means)
        if len(all_hyper_planes.size()) < 3:
            all_hyper_planes = all_hyper_planes.unsqueeze(-1)
        return all_hyper_planes, means
    def projection_metric(self, target_features, hyperplanes, mu):
        eps = 1e-12
        batch_size = target_features.shape[0]#一个batch的样本数
        class_size = hyperplanes.shape[0]#一个batch的类别数
        similarities = []
        discriminative_loss = 0.0
        for j in range(class_size):
            h_plane_j =  hyperplanes[j].unsqueeze(0).repeat(batch_size, 1, 1)#平面是2维，复制batch个同样的平面，此时是3维
            target_features_expanded = (target_features - mu[j].expand_as(target_features)).unsqueeze(-1)#expand_as将mu扩展为和target_features一致的维度,unsqueeze(-1)将特征拉成列方便下步矩阵运算
            projected_query_j = torch.bmm(h_plane_j, torch.bmm(torch.transpose(h_plane_j, 1, 2), target_features_expanded))#批量矩阵运算对每个样本在每个类空间上的投影
            projected_query_j = torch.squeeze(projected_query_j) + mu[j].unsqueeze(0).repeat(batch_size, 1)#squeeze将维度为1的数据压缩，平均向量1维，复制batch个是2维
            projected_query_dist_inter = target_features - projected_query_j
            #Training per epoch is slower but less epochs in total
            query_loss = -torch.sqrt(torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) + eps)#norm||.||每个样本到空间j的距离组成行向量
            #Training per epoch is faster but more epochs in total
            #query_loss = -torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) # Squared norm ||.||^2
            similarities.append(query_loss)#1维行向量按行堆叠成2维矩阵,行数为batch的类数,列数为batch的样本数
            for k in range(class_size):
                if j != k:
                   temp_loss = torch.mm(torch.transpose(hyperplanes[j], 0, 1), hyperplanes[k]) ## discriminative subspaces (Conv4 only, ResNet12 is computationally expensive)
                   discriminative_loss = discriminative_loss + torch.sum(temp_loss*temp_loss)#torch.sum不指定维度是默认所有元素相加
        similarities = torch.stack(similarities, dim=1)#最终行数为为batch的类数，列为样本数;tensor没有append,先列表append再stack转为tensor列表
        return similarities, discriminative_loss

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),#必须在激活函数之前BN
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):#输入输出通道数
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)#每一个样本的特征为行向量

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-workers',default=1,type=int)
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=1)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='XJTU_Conv4/models/DsnNet')
    # parser.add_argument('--data-path', default='your miniimagenet folder')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lamb', type=float, default=0.005)

    args = parser.parse_args()
    args.subspace_dim = args.shot-1
    pprint(vars(args))
    set_gpu(args.gpu)

    model = ConvNet().cuda()
    if args.shot == 1:
        shot_num = 2
        args.subspace_dim = 1
    else:
        shot_num = args.shot
    projection_pro = Subspace_Projection(num_dim=args.subspace_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.shot > 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    trlog = {}
    trlog['args'] = vars(args)#指令加入字典
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_valid_acc'] = 0.0

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label,100,args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset,batch_sampler=train_sampler,num_workers=1, pin_memory=False)#train_loader既含数据又含标签

    valset = MiniImageNet('valid')
    val_sampler = CategoriesSampler(valset.label,100,args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset,batch_sampler=val_sampler,num_workers=1, pin_memory=False)
    for epoch in range(1, args.max_epoch + 1):#range(args.max_epoch)默认从0到max_epoch-1,该行则是1到max_epoch
        model.train()#进入训练模式
        tl = Averager()
        ta = Averager()
        for i, t_batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in t_batch]#batch既含数据又含标签,训练只用到了标签
            p = args.shot * args.train_way
            qq = p + args.query * args.train_way
            data_shot, data_query = data[:p], data[p:qq]

            if args.shot == 1:
                data_shot = torch.cat((data_shot, flip(data_shot, 3)), dim=0)#按行拼接，类似于stack（）

            proto = model(data_shot) #batch输出特征按[[1类1样本:1xM特征],[2类1样本:1xM],[1类2样本:1xM],[2类2样本:1xM]]排列(2维),行数为N类xk样本数
            proto = proto.reshape(shot_num, args.train_way, -1) #batch输出变为[[[1类1样本:1xM],[2类1样本:1xM]],[[1类2样本:1xM],[2类2样本:1xM]]](3维)
            proto = torch.transpose(proto, 0, 1) #交换行和列的维度:batch输出变为[[[1类1样本:1xM],[1类2样本:1xM]],[[2类1样本:1xM],[2类2样本:1xM]]](3维)
            hyperplanes, mu = projection_pro.create_subspace(proto, args.train_way, shot_num)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits, discriminative_loss = projection_pro.projection_metric(model(data_query), hyperplanes, mu=mu)
            loss = F.cross_entropy(logits, label) + args.lamb*discriminative_loss
            acc = count_acc(logits, label)

            tl.add(loss.item())#item()将tensor转为标量
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        tl = tl.item()#n个batch的平均值
        ta = ta.item()
        print('epoch {}, loss={:.4f} acc={:.4f}'.format(epoch, loss.item(), acc))

        if epoch < 5 and epoch%2!=0:
            continue
        model.eval()#进入验证模式，此时训练模式暂停
        vl = Averager()
        v_va = []
        for i, v_batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in v_batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            if args.shot == 1:
                data_shot = torch.cat((data_shot, flip(data_shot, 3)), dim=0)

            proto = model(data_shot)
            proto = proto.reshape(shot_num, args.test_way, -1) ## change to two samples num_shot=2 with flipped one if shot=1
            proto = torch.transpose(proto, 0, 1)
            hyperplanes,  mu = projection_pro.create_subspace(proto, args.test_way, shot_num)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits, _ = projection_pro.projection_metric(model(data_query), hyperplanes, mu=mu)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            v_va.append(acc)

        vl = vl.item()
        va, h = mean_confidence_interval(v_va)
        print('epoch {}, val, loss={:.4f} acc={:.4f} h={:.4f} maxacc={:.4f}'.format(epoch, vl, va, h, trlog['max_valid_acc']))

        if va > trlog['max_valid_acc']:
            trlog['max_valid_acc'] = va
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(),str(args.save_path+"/feature_encoder_"+ str(args.train_way) + '_way_' + str(args.shot) + '_shot Max_acc.pkl')) #保存验证精度最高的模型参数
        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(trlog, osp.join(args.save_path, 'trlog'))
        # print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))#显示平均每次迭代的时间
    

    testset = MiniImageNet('test')
    test_sampler = CategoriesSampler(testset.label,600,args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset,batch_sampler=test_sampler,num_workers=1, pin_memory=False)

    model1 = ConvNet().cuda()
    model1.load_state_dict(torch.load(str(args.save_path+"/feature_encoder_"+ str(args.train_way) + '_way_' + str(args.shot) + '_shot Max_acc.pkl')))

    projection_pro = Subspace_Projection(num_dim=args.subspace_dim)

    te_l = Averager()
    t_te_a = []
    for i, batch in enumerate(test_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.test_way
        data_shot, data_query = data[:p], data[p:]

        if args.shot == 1:
            data_shot = torch.cat((data_shot, flip(data_shot, 3)), dim=0)

        proto = model1(data_shot)
        proto = proto.reshape(shot_num, args.test_way, -1) ## change to two samples num_shot=2 with flipped one if shot=1
        proto = torch.transpose(proto, 0, 1)
        hyperplanes,  mu = projection_pro.create_subspace(proto, args.test_way, shot_num)

        label = torch.arange(args.test_way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        logits, _ = projection_pro.projection_metric(model1(data_query), hyperplanes, mu=mu)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        te_l.add(loss.item())
        t_te_a.append(acc)

    te_l = te_l.item()
    te_a, h = mean_confidence_interval(t_te_a)
    print(' TEST loss={:.4f} acc={:.4f} h={:.4f}'.format(te_l, te_a, h))