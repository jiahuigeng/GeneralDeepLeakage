import time

import argparse
import os.path as osp
from torchvision import datasets, transforms

from src.models import *
# from datasets.data import get_dataset
from src.data import *
from src.logger import *
from src.saver import *
from src.utils import *


def get_grad_diff(dummy_data, original_dy_dxs, net, label_pred, criterion):
    dummy_data_copy = copy.deepcopy(dummy_data)
    original_dy_dxs_copy = copy.deepcopy(original_dy_dxs)
    net = copy.deepcopy(net)
    net.zero_grad()
    pred, _ = net(dummy_data_copy)
    dummy_loss = criterion(pred, label_pred)
    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

    grad_diff = 0
    for original_dy_dx in original_dy_dxs_copy:
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
    return grad_diff



def gradient_clousure(optimizer, dummy_data, original_dy_dxs, original_nets, dummy_label, label_pred, method, criterion, tv_alpha=1000, clip_alpha=100, scale_alpha=100, l2_alpha=0):
    torch.manual_seed(7)
    def closure():
        optimizer.zero_grad()
        l2_loss = torch.square(dummy_data).detach()
        clip_loss = torch.square(dummy_data - torch.clip(dummy_data, 0, 1)).detach()

        dummy_max, dummy_min = torch.max(dummy_data), torch.min(dummy_data)
        scale = (dummy_max - dummy_min)
        scale_loss = torch.square((dummy_data - dummy_min) / scale- dummy_data).detach()

        grad_diff = 0
        for net, dy_dx in zip(original_nets, original_dy_dxs):
            pred, _ = net(dummy_data)
            if method == "dlg":
                dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
            else:
                dummy_loss = criterion(pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            for dummy_gl, gl in zip(dummy_dy_dx, dy_dx):
                # grad_diff += ((dummy_gl - gl) **2).sum()
                grad_diff += 1 - torch.nn.functional.cosine_similarity(dummy_gl.flatten(), gl.flatten(), 0, 1e-10)
            grad_diff += tv_alpha * total_variation(dummy_data) + l2_alpha * torch.sum(l2_loss) \
                     + scale_alpha * torch.sum(scale_loss) + clip_alpha * torch.sum(clip_loss)

        grad_diff.backward()
        return grad_diff / len(original_dy_dxs)

        # pred, _ = net(ori_dummy_data)
        # if method == 'dlg':
        #     dummy_loss = - torch.mean(
        #         torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
        # else:
        #     dummy_loss = criterion(pred, label_pred)
        #
        # dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        #
        # grad_diff = 0
        # for original_dy_dx in original_dy_dxs:
        #     for gx, gy in zip(dummy_dy_dx, original_dy_dx):
        #         grad_diff += ((gx - gy) ** 2).sum()
        #     grad_diff += tv_alpha * total_variation(dummy_data) + l2_alpha * torch.sum(l2_loss) \
        #                  + scale_alpha * torch.sum(scale_loss) + clip_alpha * torch.sum(clip_loss)
        # grad_diff.backward()
        # return grad_diff/len(original_dy_dxs)
    return closure

def train_steps(model, data, label, device, args):
    model = copy.deepcopy(model)
    # local_bs = args.local_bs
    local_bs = args.bs
    local_epochs = args.local_epochs
    local_lr = args.local_lr

    torch.manual_seed(7)
    # model_clone = copy.deepcopy(model)
    model.zero_grad()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), local_lr)

    for e in range(local_epochs):
        for i in range(data.shape[0] // local_bs):
            optimizer.zero_grad()
            idx = i % (data.shape[0] // local_bs)
            outputs, _ = model(data[idx * local_bs: (idx + 1) * local_bs])
            labels = label[idx * local_bs: (idx+1) * local_bs]
            loss = criterion(outputs, labels)
            logger.info("loss in train steps: %s" % loss)
            loss.backward()
            optimizer.step()
    return model

def main(args):

    lr = args.lr
    bs = args.bs
    Iteration = args.Iteration
    np.random.seed(7)
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    save_dir_name = args.task + '_' + args.dataset + '_' + args.model + '_' + args.mode + '_bs' + str(args.bs) + '_ds' + \
        str(args.data_scale) + '_tv' + str(args.tv_alpha) + '_l2' + str(args.l2_alpha) + '_sc' +str(args.scale_alpha) + '_cl' + str(args.clip_alpha) + \
                    '_lr' + str(args.lr) +'_ld' + str(args.load_model) + '_' + args.exp_name

    if not osp.exists("results"):
        os.mkdir("results")


    save_dir = osp.join("results", save_dir_name)
    if not osp.exists(save_dir):
        os.mkdir(osp.join("results", save_dir_name))
    runs = sorted(glob.glob(osp.join(save_dir, 'experiment_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    experiment_dir = osp.join(save_dir, 'experiment_{}'.format(str(run_id)))
    if not osp.exists(experiment_dir):
        os.mkdir(experiment_dir)

    logger = get_logger(experiment_dir)
    saver = Saver(experiment_dir)

    logger.info(args)
    logger.info("Experiment Dir: %s" % experiment_dir)


    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    dst, channel, num_classes, hidden = get_dataset(args)
    net = get_model(channel=channel, hidden=hidden, num_classes=num_classes, args=args)

    num_exp = min(int(len(dst)/bs), args.num_exp)
    start_idx = 0
    index = get_index(args.dataset)


    for idx_net in range(num_exp):
        total_label_pred = dict()
        total_image_restore = dict()
        attack_list = ["dlg"] # "dlg", "gradinversion",
        # if bs > num_classes:
        #     attack_list = ["dlg", "gdlg"]
        for method in attack_list:
            total_label_pred[method] = list()
            total_image_restore[method] = list()

        logger.info('running %d|%d experiment' % (idx_net, num_exp))
        net = net.to(device)

        gt_data, gt_label, gt_index = None, None, None
        idx_shuffle = np.random.permutation(len(dst))
        if args.data_mode == "random":
            gt_index = list(range(start_idx, start_idx+bs))
            gt_data = [tt(dst[idx_shuffle[start_idx+i]][0]).unsqueeze(0).float().to(device) for i in range(bs)]
            gt_label = [torch.Tensor([dst[idx_shuffle[start_idx+i]][1]]).long().to(device) for i in range(bs)]
            start_idx += bs

        elif args.data_mode in ["repeat", "repeat_label"]:
            cls_idx = int(start_idx % (num_classes*bs)/bs)
            sub_idx = int(start_idx / num_classes / bs) * bs
            gt_index = index[cls_idx][sub_idx: sub_idx+bs]
            gt_data = [tt(dst[i][0]).unsqueeze(0).float().to(device) for i in gt_index]
            gt_label = [torch.Tensor([dst[i][1]]).long().to(device) for i in gt_index]
            start_idx += bs


        elif args.data_mode in ["repeat_image"]:
            cls_idx = int(start_idx / num_classes)
            sub_idx = int(start_idx % num_classes)
            gt_index = index[cls_idx][sub_idx]
            gt_data = [tt(dst[gt_index][0]).unsqueeze(0).float().to(device) for _ in range(bs)]
            gt_label = [torch.Tensor([dst[gt_index][1]]).long().to(device) for _ in range(bs)]
            start_idx += 1

        elif args.data_mode == "unique":
            gt_index = []
            for idx in range(bs):
                cls_idx = (start_idx + idx) %num_classes
                sub_idx = int((start_idx + idx) / num_classes)
                gt_index.append(index[cls_idx][sub_idx])
            start_idx += bs
            gt_data = [tt(dst[i][0]).unsqueeze(0).float().to(device) for i in gt_index]
            gt_label = [torch.Tensor([dst[i][1]]).long().to(device) for i in gt_index]

        # factor: [11223344]
        elif args.data_mode == "factor":
            gt_index = []
            repeat = args.repeat

            for idx in range(bs):
                cls_idx = int((start_idx + idx)/repeat) % num_classes
                already = int((start_idx + idx)/repeat /num_classes)
                sub_idx = already + (start_idx + idx) - already * repeat * num_classes  - cls_idx * repeat

                gt_index.append(index[cls_idx][sub_idx])
            start_idx += bs
            gt_data = [tt(dst[i][0]).unsqueeze(0).float().to(device) for i in gt_index]
            gt_label = [torch.Tensor([dst[i][1]]).long().to(device) for i in gt_index]


        # single: [11112345]
        elif args.data_mode == "single":
            gt_index = []
            repeat = args.repeat
            for idx in range(bs - repeat + 1):
                cls_idx = (start_idx + idx) % num_classes
                sub_idx = int((start_idx + idx) / num_classes)
                gt_index.append(index[cls_idx][sub_idx])
            cls_idx = start_idx % num_classes
            sub_idx = int(start_idx / num_classes) + 1
            for idx in range(repeat - 1):
                gt_index.append(index[cls_idx][sub_idx + idx])
            start_idx += bs - repeat + 1
            gt_data = [tt(dst[i][0]).unsqueeze(0).float().to(device) for i in gt_index]
            gt_label = [torch.Tensor([dst[i][1]]).long().to(device) for i in gt_index]



        gt_data = torch.cat(gt_data, dim=0)
        gt_label = torch.cat(gt_label, dim=0)
        logger.info("gt_index: %s; gt_label: %s" % (str(gt_index), str(gt_label)))



        for method in attack_list: # model
            logger.info('%s, Try to generate %d images' % (method, bs))
            if args.model == "lenet":
                net.apply(weights_init)
            criterion = nn.CrossEntropyLoss().to(device)


            # compute original gradient

            original_dy_dxs = []
            original_nets = []
            if args.mode == "gradients":

                out, _ = net(gt_data)
                y = criterion(out, gt_label)
                tmp_dy_dx = torch.autograd.grad(y, net.parameters())
                original_dy_dxs.append(list((_.detach().clone() for _ in tmp_dy_dx)))
                original_nets.append(copy.deepcopy(net))
                for i in range(1, args.multi_steps):
                    net = train_steps(net, gt_data, gt_label, device, args)
                    out, _ = net(gt_data)
                    y = criterion(out, gt_label)
                    tmp_dy_dx = torch.autograd.grad(y, net.parameters())
                    original_dy_dxs.append(list((_.detach().clone() for _ in tmp_dy_dx)))
                    original_nets.append(copy.deepcopy(net))




                # local_model = copy.deepcopy(net)
                # local_out, _ = local_model(gt_data)
                # local_y = criterion(local_out, gt_label)
                # tmp_dy_dx = torch.autograd.grad(local_y, local_model.parameters())
                # original_dy_dxs.append(list((_.detach().clone() for _ in tmp_dy_dx)))
                # original_models.append(local_model)
                #
                # for i in range(args.multi_steps):
                #     # local_model = copy.deepcopy(local_model)
                #     local_model = train_steps(local_model, gt_data, gt_label, device, args)
                #     local_out, _ = local_model(gt_data)
                #     local_y = criterion(local_out, gt_label)
                #     tmp_dy_dx = torch.autograd.grad(local_y, local_model.parameters())
                #     original_dy_dxs.append(list((_.detach().clone() for _ in tmp_dy_dx)))


            elif args.mode == "weights":
                init_model = copy.deepcopy(net)
                local_model = copy.deepcopy(net)
                for i in range(args.multi_steps):
                    local_model = train_steps(local_model, gt_data, gt_label, device, args)
                    cur_diff = [(l_pre.detach() - l_cur.detach())/args.local_lr for l_pre, l_cur in zip(init_model.parameters(), local_model.parameters())]
                    original_dy_dxs.append(cur_diff)


            dummy_data = torch.rand(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.rand((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
            dummy_data_copy = copy.deepcopy(dummy_data)

            optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)

            label_pred = []
            if method == 'dlg':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
            elif method == "gradinversion":
                # tmp_min = torch.sum(original_dy_dxs[0][-2], dim=-1)
                tmp_min = torch.min(original_dy_dxs[0][-2], dim=-1).values
                label_pred = torch.argsort(tmp_min).detach()[:bs]
                logger.info("gradinversion label prediction: %s", str(label_pred))


            elif method in  ["gdlg"]:
                tmp_min = torch.sum(original_dy_dxs[0][-2], dim=-1)
                dummy_z, dummy_o = net(dummy_data_copy)
                dummy_z, dummy_o = dummy_z.detach().clone(), dummy_o.detach().clone()
                dummy_p = F.softmax(dummy_z, dim=-1)
                sum_p = torch.sum(dummy_p, dim=0)
                sum_o = torch.sum(torch.mean(dummy_o, dim=0))
                # label_pred = sum_p - (bs / sum_o) * tmp_min
                label_p = (sum_p - (bs / sum_o) * tmp_min).detach().clone().cpu().numpy()
                reds = np.round(label_p)
                tmp_mods = np.abs(label_p - reds)
                mods = []
                # mods = np.array([mod for mod in mods if mod <= 0 else mod/res])
                for id, mod in enumerate(tmp_mods):
                    if reds[id] >=1:
                        mods.append(mod/reds[id])
                    else:
                        mods.append(mod)
                label_pred = []
                for idx in np.argsort(mods):
                    if reds[idx] > 0:
                        label_pred += [idx for _ in range(int(reds[idx]))]
                    if len(label_pred )> bs:
                        break
                if len(label_pred) < bs:
                    mods = torch.tensor(mods)
                    for idx in torch.argsort(mods, descending=True):
                        if idx not in label_pred:
                            label_pred += [idx]
                        if len(label_pred) > bs:
                            break
                label_pred = label_pred[:bs]
                label_pred = torch.tensor(label_pred).to(device)
                logger.info("gdlg label prediction: %s" % str(label_pred))


            if args.infer_mode == "data":
                label_pred = gt_label
                # if args.mislabel > 0:
                #     pass
                # if args.

            if args.infer_mode in ["both", "data"]:
                history = []
                history_iters = []
                losses = []
                mses = []
                train_iters = []
                result_imgs = []

                logger.info('lr = %s', str(lr))
                if args.skip:
                    Iteration = 0
                for iters in range(Iteration):
                    closure = gradient_clousure(optimizer, dummy_data, original_dy_dxs, original_nets, dummy_label, label_pred,
                                                method, criterion, tv_alpha=args.tv_alpha, clip_alpha=args.clip_alpha,
                                                scale_alpha=args.scale_alpha, l2_alpha=args.l2_alpha)
                    current_loss = optimizer.step(closure)

                    if np.isnan(current_loss.cpu().detach().numpy()):
                        break

                    train_iters.append(iters)
                    losses.append(current_loss)
                    mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                    if iters % int(Iteration / 5) == 0:
                        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                        logger.info('%s, %d, loss = %.8f, mse = %.8f' % (current_time, iters, current_loss, mses[-1]))
                        result_imgs.append(dummy_data.detach().clone().cpu())
                        history.append([tp(dummy_data[imidx].cpu()) for imidx in range(bs)])
                        history_iters.append(iters)

                        if current_loss < 0.000001:  # converge
                            break

                result_imgs.append(gt_data.detach().clone().cpu())

                if method == "dlg":
                    label_pred = torch.argmax(dummy_label, dim=-1).detach()

                gradients_diff = get_grad_diff(dummy_data, original_dy_dxs, net, label_pred, criterion)

                if args.skip:
                    label_acc = label_align(gt_label, label_pred)
                    total_sim = dict()
                    for metric in ["mse", "psnr", "ssim", "lpips"]:
                        total_sim[metric] = "0"

                else:
                    try:
                        gt_data_copy, gt_label_copy, dummy_data_copy, label_pred_copy, total_sim, label_acc, pairs = image_align(
                            gt_data, gt_label, dummy_data, label_pred, "psnr")
                    except:
                        continue

                logger.info(str(total_sim))
                total_image_restore[method].append(total_sim)
                total_label_pred[method].append(label_acc)
                gradients_diff = float(gradients_diff.data)
                if not args.skip:
                    saver.save_result_imgs(result_imgs, gt_label.detach().clone().cpu(), start_idx, method)
                # pairs = np.argsort(pairs)
                for i in range(len(result_imgs)-1):
                    result_imgs[i] = torch.index_select(result_imgs[i], 0, torch.tensor(pairs).long())
                # result_imgs_reorder = [torch.index_select(item, 0, torch.tensor(pairs).long()) for item in result_imgs]
                saver.save_result_imgs(result_imgs, gt_label.detach().clone().cpu(), start_idx, method+"_align")
                saver.save_metrics1(start_idx, gt_label=gt_label.detach().clone().cpu(),
                                    pred_label=label_pred.detach().clone().cpu(), label_acc=label_acc / bs,
                                    grad_diff=gradients_diff, mse=total_sim['mse'], ssim=total_sim['ssim'],
                                    psnr=total_sim['psnr'], lpips=total_sim['lpips'], method=method)

            elif args.infer_mode == "label":
                gt_data_copy, gt_label_copy, dummy_data_copy, label_pred_copy, total_sim, label_acc, pairs = image_align(
                    gt_data, gt_label, dummy_data, label_pred, "psnr")
                saver.save_metrics1(start_idx, gt_label=gt_label.detach().clone().cpu(),
                                    pred_label=label_pred.detach().clone().cpu(), label_acc=label_acc / bs,
                                    grad_diff=0, mse=0, ssim=0,
                                    psnr=0, lpips=0, method=method)
        logger.info('----------------------\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument('--infer-mode', type=str, default="both", choices=['data', 'label', 'both'])
    parser.add_argument('--mislabel', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='imagen', choices=['imagen', 'cifar10', 'cifar100', 'lfw'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'resnet'],
                        help='model name (default: lenet)')
    parser.add_argument('--load-model', type=bool, default=False, help='whether to use trained model (default: False)')
    parser.add_argument('--model-path', type=str, default=None, help='path to the trained model (default: False)')
    parser.add_argument('--mode', type=str, default='gradients', choices=['gradients', 'weights'],
                        help='method name (default: gradients)')
    parser.add_argument('--task', type=str, default='image', choices=['image', 'text'],
                        help='restoration task name (default: image)')
    parser.add_argument('--num-seeds', type=int, default=5, help='number of seeds for dummy image initialization')
    parser.add_argument('--bs', type=int, default=1, help='number of dummy images in a batch (batch size)')
    parser.add_argument('--data-init', type=str, default="uniform")
    parser.add_argument('--data-scale', type=int, default=3, choices=[0,1,2,3,4])
    parser.add_argument('--data-mode', type=str, default="unique", choices=['random', 'repeat', "unique", "single", "factor", "repeat_label", "repeat_image"])
    parser.add_argument('--local-lr', type=float, default=0.1)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--local-bs', type=int, default=1)
    parser.add_argument("--multi-steps", type=int, default=0)
    parser.add_argument('--dummy-norm', type=str, default="scale", choices=['clip', 'scale'])
    parser.add_argument('--tv-alpha', type=float, default=200)
    parser.add_argument('--clip-alpha', type=float, default=200)
    parser.add_argument('--scale-alpha', type=float, default=200)
    parser.add_argument('--l2-alpha', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--Iteration', type=int, default=600)
    parser.add_argument('--num-exp', type=int, default=500)
    parser.add_argument('--exp-name', type=str, default="", required=True)
    parser.add_argument('--skip', type=bool, default=False)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()


    args.cuda = torch.cuda.is_available()

    logger.info(args)

    main(args)
