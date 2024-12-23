import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import datetime
from torch.autograd import Variable
import random
import pdb
import math
from utils import ReplayBuffer,weights_init_normal
import itertools
import net
from read_data import ChestXrayDataSet,Chest
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

SOURCE_IMAGE_LIST = './data/source/labels/train_list.txt'
TEST_IMAGE_LIST = './data/source/labels/test_list.txt'


def image_classification_test(loader, model, classifier2,test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    features_test, predict_out = model(inputs[j])
                    predict_out = classifier2(features_test)
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float().cpu()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                features_test, outputs = model(inputs)
                outputs = classifier2(features_test)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float().cpu()), 0)
    _, predict = torch.max(all_output, 1)
    _,all_label = torch.max(all_label, 1)
    predict_num = torch.zeros((1, 2))
    pre_mask = torch.zeros(all_output.size()).scatter_(1, predict.cpu().view(-1, 1), 1.)
    predict_num += pre_mask.sum(0)
    tar_mask = torch.zeros(all_output.size()).scatter_(1, all_label.cpu().view(-1, 1), 1.)
    target_num = tar_mask.sum(0)
    acc_mask = pre_mask * tar_mask
    acc_num = acc_mask.sum(0)  # 得到各类别分类正确的样本数量
    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    sensitivity = acc_mask.sum(0) / tar_mask.sum(0)
    accuracy_1 = 100. * acc_num.sum(-1) / target_num.sum(-1)

    auc = roc_auc_score(all_label,all_output[:,1])

    accuracy = torch.sum(predict.float() == all_label).item() / float(all_label.size()[0])

    return accuracy,accuracy_1,precision,F1,recall,sensitivity,auc


def train(config):

    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])


    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] =   ChestXrayDataSet(data_dir=data_config["source"]["list_path"],
                     image_list_file=SOURCE_IMAGE_LIST,
                     len_=config["len"],
                     transform=prep_dict["source"])

    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)
    dsets["target"] =Chest(data_dir=data_config["target"]["list_path"],
          transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [Chest(data_dir=data_config["test"]["list_path"],
                            modal='test',transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=True, num_workers=0) for dset in dsets['test']]
    else:
        dsets["test"] = Chest(data_dir=data_config["test"]["list_path"],
                                modal='test',transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=True, num_workers=8, pin_memory=True)


    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## 添加判别器D_s,D_t,生成器G_s2t,G_t2s

    z_dimension = 256
    D_s = network.models["Discriminator"]()
    D_s = D_s.cuda()
    G_s2t = network.models["Generator"](z_dimension, 1024)
    G_s2t = G_s2t.cuda()
    de_ = network.models["Decoder"]()
    de_ = de_.cuda()

    D_t = network.models["Discriminator"]()
    D_t = D_t.cuda()
    G_t2s = network.models["Generator"](z_dimension, 1024)
    G_t2s = G_t2s.cuda()

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_Sem = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(G_s2t.parameters(), G_t2s.parameters()), lr=0.0003)
    optimizer_D_s = torch.optim.Adam(D_s.parameters(), lr=0.0003)
    optimizer_D_t = torch.optim.Adam(D_t.parameters(), lr=0.0003)

    fake_S_buffer = ReplayBuffer()
    fake_T_buffer = ReplayBuffer()

    classifier_optimizer = torch.optim.Adam(base_network.parameters(), lr=0.0003)
    ## 添加分类器
    classifier1 = net.Net(256,class_num)
    classifier1 = classifier1.cuda()
    classifier1_optim = optim.Adam(classifier1.parameters(), lr=0.0003)
    classifier2 = net.Net(256, 2)
    classifier2 = classifier2.cuda()
    classifier2_optim = optim.Adam(classifier2.parameters(), lr=0.0003)

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        base_network = base_network.module

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    len_test_target = len(dset_loaders["test"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    temp_acc = 0.0
    best_iter = 0.0

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc, accuracy_1, precision, F1, recall, sensitivity, auc = image_classification_test(dset_loaders,
                                                                                                      base_network,
                                                                                                      classifier2,
                                                                                                      test_10crop=
                                                                                                      prep_config[
                                                                                                          "test_10crop"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
                best_iter = i
                best_accuracy_1 = accuracy_1
                best_precision = precision
                best_F1 = F1
                best_recal = recall
                best_sensitivity = recall
                best_auc = auc

                now = datetime.datetime.now()
                d = str(now.month) + '-' + str(now.day) + ' ' + str(now.hour) + ':' + str(now.minute) + ":" + str(
                    now.second)
                torch.save(base_network.state_dict(), osp.join(config["output_path"],
                                                "{}_to_{}_best_model_acc-{}_{}.pth.tar".format(args.source, args.target,
                                                                                               best_acc, d)))
                torch.save(classifier2.state_dict(), osp.join(config["output_path"],
                                                               "{}_to_{}_classfi_acc-{}_{}.pth.tar".format(
                                                                   args.source, args.target,
                                                                   best_acc, d)))
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)

            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()

            print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(base_network.state_dict(), osp.join(config["output_path"], \
                                                             "{}_to_{}_iter_{:05d}_model_{}.pth.tar".format(args.source,
                                                                                                            args.target,
                                                                                                            i, str(
                                                                     datetime.datetime.utcnow()))))

        loss_params = config["loss"]
        ## train one iter
        classifier1.train(True)
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()


        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target,labels_source,labels_target = inputs_source.cuda(), inputs_target.cuda(),\
                                                    labels_source.cuda(), labels_target.cuda()
        # 提取特征
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        outputs_source1 = classifier1(features_source.detach())
        outputs_target1 = classifier1(features_target.detach())
        outputs1 = torch.cat((outputs_source1,outputs_target1),dim=0)
        softmax_out1 = nn.Softmax(dim=1)(outputs1)


        softmax_out = (1-args.cla_plus_weight)*softmax_out + args.cla_plus_weight*softmax_out1

        if config['method'] == 'CDAN+E':
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method'] == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        elif config['method'] == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')

        classifier_loss = nn.BCEWithLogitsLoss()(outputs_source, labels_source )


        # Cycle
        num_feature = features_source.size(0)
        # =================train discriminator T
        real_label = Variable(torch.ones(num_feature)).cuda()
        fake_label = Variable(torch.zeros(num_feature)).cuda()

        # 训练生成器
        optimizer_G.zero_grad()

        # Identity loss
        same_t = G_s2t(features_target.detach())
        loss_identity_t = criterion_identity(same_t, features_target)

        same_s = G_t2s(features_source.detach())
        loss_identity_s = criterion_identity(same_s, features_source)

        # Gan loss
        fake_t = G_s2t(features_source.detach())

        pred_fake = D_t(fake_t)
        loss_G_s2t = criterion_GAN(pred_fake, labels_source.float())

        fake_s = G_t2s(features_target.detach())
        pred_fake = D_s(fake_s)
        loss_G_t2s = criterion_GAN(pred_fake, labels_source.float())

        # cycle loss
        recovered_s = G_t2s(fake_t)
        loss_cycle_sts = criterion_cycle(recovered_s, features_source)

        recovered_t = G_s2t(fake_s)
        loss_cycle_tst = criterion_cycle(recovered_t, features_target)

        # sem loss
        pred_recovered_s = base_network.fc(recovered_s)
        pred_fake_t = base_network.fc(fake_t)
        loss_sem_t2s = criterion_Sem(pred_recovered_s, pred_fake_t)

        pred_recovered_t = base_network.fc(recovered_t)
        pred_fake_s = base_network.fc(fake_s)
        loss_sem_s2t = criterion_Sem(pred_recovered_t, pred_fake_s)

        loss_cycle = loss_cycle_tst + loss_cycle_sts
        weights = args.weight_in_lossG.split(',')
        loss_G = float(weights[0]) * (loss_identity_s + loss_identity_t) + \
                 float(weights[1]) * (loss_G_s2t + loss_G_t2s) + \
                 float(weights[2]) * loss_cycle + \
                 float(weights[3]) * (loss_sem_s2t + loss_sem_t2s)



        # 训练softmax分类器
        outputs_fake = classifier1(fake_t.detach())
        
        # 分类器优化
        classifier_loss1 = nn.BCEWithLogitsLoss()(outputs_fake, labels_source)
        classifier1_optim.zero_grad()
        classifier_loss1.backward()
        classifier1_optim.step()
        

        outputs_target2 = classifier2(features_target.detach())
        classifier_loss_ = nn.BCEWithLogitsLoss()(outputs_target2, labels_target)
        classifier2_optim.zero_grad()
        classifier_loss_.backward()
        classifier2_optim.step()


        #total_loss = loss_params["trade_off"] * transfer_loss + args.class_weight*classifier_loss +  args.cyc_loss_weight*loss_G
        total_loss = args.class_weight*classifier_loss + loss_params["trade_off"] * transfer_loss + loss_cycle * 0.0005
        total_loss.backward()
        optimizer.step()
        classifier_optimizer.step()

        
        optimizer_G.step()

        ###### Discriminator S ######
        optimizer_D_s.zero_grad()

        # Real loss
        pred_real = D_s(features_source.detach())
        loss_D_real = criterion_GAN(pred_real, real_label)

        # Fake loss
        fake_s = fake_S_buffer.push_and_pop(fake_s)
        pred_fake = D_s(fake_s.detach())
        loss_D_fake = criterion_GAN(pred_fake, fake_label)

        # Total loss
        loss_D_s = loss_D_real + loss_D_fake
        loss_D_s.backward()

        optimizer_D_s.step()
        ###################################

        ###### Discriminator t ######
        optimizer_D_t.zero_grad()

        # Real loss
        pred_real = D_t(features_target.detach())
        loss_D_real = criterion_GAN(pred_real, real_label)

        # Fake loss
        fake_t = fake_T_buffer.push_and_pop(fake_t)
        pred_fake = D_t(fake_t.detach())
        loss_D_fake = criterion_GAN(pred_fake, fake_label)

        # Total loss
        loss_D_t = loss_D_real + loss_D_fake
        loss_D_t.backward()
        optimizer_D_t.step()

        if i % 50 == 0:
            print(f"Epoch [{i + 1}/{2000}] - total_loss: {total_loss:.4f} - loss_D_t: {loss_D_t:.4f}")
    now = datetime.datetime.now()
    d = str(now.month)+'-'+str(now.day)+' '+str(now.hour)+':'+str(now.minute)+":"+str(now.second)
    torch.save(base_network.state_dict(), osp.join(config["output_path"],
                                    "{}_to_{}_best_model_acc-{}_{}.pth.tar".format(args.source, args.target,
                                                                            best_acc,d)))
    print(f"Best_acc: {best_acc:.4f},Best_iter: {best_iter:.4f}")
    print(
        "best_accuracy_1: ", best_accuracy_1, "best_precision: ", best_precision,
        "best_F1:", best_F1, "best_recal: ", best_recal, "best_sensitivity: ", best_sensitivity, "best_auc:", best_auc)
    return best_accuracy_1, best_precision, best_F1, best_recal, best_auc


if __name__ == "__main__":

    torch.cuda.manual_seed(42)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--net', type=str, default="VGG11",
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='NIH', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='./data/source/images-224/images-224',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='./data/target/train',
                        help="The target dataset path list")
    parser.add_argument('--t_test_path', type=str, default='./data/target/test',
                        help="The target dataset path list")
    parser.add_argument('--source', type=str, default="CHEST", help="The source dataset name")
    parser.add_argument('--target', type=str, default="NIH", help="The target dataset name")
    parser.add_argument('--test_interval', type=int, default=100, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--cyc_loss_weight',type=float,default=0.005)
    parser.add_argument('--cla_plus_weight', type=float, default=0.1)
    parser.add_argument("--weight_in_lossG",type=str,default='1,0.01,0.1,0.1')
    parser.add_argument('--class_weight', type=float, default=1.0)


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id



    # train config
    task_name = args.source+"2"+args.target
    config = {}
    config['torch_seed'] = torch.initial_seed()
    config['torch_cuda_seed'] = torch.cuda.initial_seed()
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config['cyc_loss_weight'] = args.cyc_loss_weight
    config['cla_plus_weight'] = args.cla_plus_weight
    config['weight_in_lossG'] = args.weight_in_lossG
    config["num_iterations"] = 2000
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + task_name
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log_{}_to_{}_{}.txt".format(args.source, args.target,
                                                                                           str(
                                                                                               datetime.datetime.utcnow()))),
                              "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop": False, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": 1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 36}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": 36}, \
                      "test": {"list_path": args.t_test_path, "batch_size": 4}}

    if config["dataset"] == "NIH":
        config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 14
    elif config["dataset"] == "CHEST":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 2
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()

    config["len"] = 6000
    accuracy = []
    precision = []
    F1 = []
    recal = []
    auc = []

    for test_num in range(1, 11):
        print(test_num)
        best_accuracy_1, best_precision, best_F1, best_recal, best_auc = train(config)
        accuracy.append(best_accuracy_1)
        precision.append(best_precision)
        F1.append(best_F1)
        recal.append(best_recal)
        auc.append(best_auc)
    print("acu:", accuracy, "pre:", precision, "f1:", F1, "rec", recal, "auc:", auc)
