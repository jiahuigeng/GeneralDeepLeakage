import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips
from torchvision import datasets
LPIPS_model = lpips.LPIPS(net='alex')
from scipy.optimize import linear_sum_assignment

def cal_lpips(img0, img1):
    return LPIPS_model.forward(img0, img1)


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


# def evaluation(origin, restore):
#     assert origin.shape == restore.shape
#     mse = torch.mean((origin-restore)**2).detach()
#     ssim = None
#     asec = None
#     # asec = torch.sum(origin == restore)/torch.sum(torch.ones(origin.shape))
#     # print(torch.sum(origin == restore))
#     print(mse, ssim, asec)
#     return mse, ssim, asec

def count(ll, e):
    n = 0
    for item in ll:
        if item == e:
            n += 1
    return n

def label_align(list1, list2):
    list1, list2 = list(list1), list(list2)
    res = []
    for x in list1:
        if x in list2:
            res.append(x)
            list2.remove(x)
    return len(res)

def similarity(img0, img1, metric):
    v_sim = 0
    if metric == "ssim":
        v_sim = ssim(img0.permute([1, 2, 0]).detach().numpy(), img1.permute([1, 2, 0]).detach().numpy(), multichannel=True)
    if metric == "psnr":
        v_sim = psnr(img0.detach().numpy(), img1.detach().numpy())
    if metric == "lpips":
        v_sim = cal_lpips(img0, img1).detach().numpy()[0][0][0]
    if metric == "mse":
        v_sim = (img0.detach().numpy()- img1.detach().numpy())**2
    return v_sim


def image_align(gt_data, gt_label, pred_data, pred_label, metric):
    gt_label = copy.deepcopy(gt_label).detach().cpu()
    gt_data = copy.deepcopy(gt_data).detach().cpu()
    pred_data = copy.deepcopy(pred_data).detach().cpu()
    pred_label = copy.deepcopy(pred_label).detach().cpu()
    gt_label = gt_label.numpy().tolist()
    # pred_data = pred_data.detach().numpy()
    pred_label = pred_label.numpy().tolist()
    length = len(gt_label)
    # find the particular onos
    # similar_matrix = np.zeros([length, length], dtype=float)
    if metric in ["psnr", "ssim"]:
        similar_matrix = np.full([length, length], -100.0)
    else:
        similar_matrix = np.full([length, length], 100.0)
    pairs = np.zeros(length, dtype=np.int32)
    single_gt_label = [item for item in gt_label if count(gt_label, item) == 1]
    single_pred_label = [item for item in pred_label if count(pred_label, item) == 1]
    common_single = set(single_gt_label).intersection(single_pred_label)
    for item in common_single:
        idx_0 = gt_label.index(item)
        idx_1 = pred_label.index(item)
        # similar_matrix[idx_0][idx_1] = 1
        pairs[idx_0] = idx_1

    for idx_0 in range(length):
        if gt_label[idx_0] not in common_single:
            for idx_1 in range(length):
                if pred_label[idx_1] not in common_single:
                    similar_matrix[idx_0][idx_1] = similarity(gt_data[idx_0], pred_data[idx_1], metric)

    row_idx, col_idx = linear_sum_assignment(similar_matrix, metric in ["psnr", "ssim"])

    assert len(row_idx) == len(col_idx) == length
    for idx in range(length):
        if gt_label[row_idx[idx]] not in common_single:
            pairs[row_idx[idx]] = col_idx[idx]

    pred_data = torch.index_select(pred_data, 0, torch.tensor(pairs).long())
    pred_label = np.take(pred_label, pairs)

    label_acc = label_align(gt_label, pred_label)
    total_sim = dict()
    for metric in ["mse", "psnr", "ssim", "lpips"]:
        sims = [similarity(gt_data[idx], pred_data[idx], metric) for idx in range(length)]
        avg_sim = np.mean(sims)
        total_sim[metric] = avg_sim

    return gt_data,  gt_label, pred_data, pred_label, total_sim, label_acc, pairs

def l2_distance(img0, img1):
    return torch.sqrt((img0 - img1) ** 2).detach()

# def cal_lpips(img0, img1):
#     return LPIPS_model.forward(img0, img1)

def total_variation(x):
    """Anisotropic TV."""
    if len(x.shape) == 3:
        dx = torch.mean(torch.abs(x[ :, :, :-1] - x[ :, :, 1:]))
        dy = torch.mean(torch.abs(x[ :, :-1, :] - x[ :, 1:, :]))
    else:
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

def get_dummpy_data(shape, data_init):
    dummy_data = torch.empty(shape).requires_grad_(True)
    if data_init == "uniform":
        nn.init.uniform_(dummy_data)
    elif data_init == "normal":
        nn.init.normal_(dummy_data)
    elif data_init == "xavier_uniform":
        nn.init.xavier_uniform(dummy_data)
    return dummy_data

def data_index():

    for data in ["cifar10", "cifar100", "imagen"]:
        if data == "cifar10":
            with open("../data/cifar10_index.txt", "w") as f:
                dst_cifar10 = datasets.CIFAR10("../data", download=False)
                cifar10_index = [[] for _ in range(10)]
                for i in range(len(dst_cifar10)):
                    cifar10_index[dst_cifar10[i][1]].append(str(i))
                for i in range(10):
                    f.write(",".join((cifar10_index[i]))+"\n")
            print("cifar10 down")
        elif data == "cifar100":
            with open("../data/cifar100_index.txt", "w") as f:
                dst_cifar100 = datasets.CIFAR100("../data", download=True)
                cifar100_index = [[] for _ in range(100)]
                for i in range(len(dst_cifar100)):
                    cifar100_index[dst_cifar100[i][1]].append(str(i))
                for i in range(100):
                    f.write(",".join((cifar100_index[i]))+"\n")
            print("cifar100 down")
        elif data == "imagen":
            with open("../data/imagen_index.txt", "w") as f, open("../data/imagen/samples.txt") as f1:
                imagen_index = [[] for i in range(200)]
                for i, line in enumerate(f1):
                    line = line.split("\t")
                    label = line[1]
                    imagen_index[int(label)].append(str(i))
                for i in range(200):
                    f.write(",".join(imagen_index[i])+'\n')
            print("imagen down")


def get_index(dataset):
    index = []
    if dataset == "cifar10":
        with open("data/cifar10_index.txt") as f:
            for line in f:
                index.append(list(map(int, line.split(","))))
    elif dataset == "cifar100":
        with open("data/cifar100_index.txt") as f:
            for line in f:
                index.append(list(map(int, line.split(","))))
    elif dataset == "imagen":
        with open("data/imagen_index.txt") as f:
            for line in f:
                index.append(list(map(int, line.split(","))))

    return index



if __name__ == "__main__":
    data_index()