import os
import copy
import torch
import torch.nn as nn
import numpy as np
import random
from models.vpt_structure import build_promptmodel
from datetime import datetime
from iopath.common.file_io import PathManager as PathManagerBase
PathManager = PathManagerBase()

from tqdm import tqdm
from options import args_parser
from utils import data_utils, log_utils, update, fedavg, gpa

time_folder = datetime.now().strftime("%m%d%Y_%H%M%S")
print(time_folder)
output_folder = os.path.join('result_log', time_folder)
PathManager.mkdirs(output_folder)


def setup_seed(seed=1):  # setting up the random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    np.set_printoptions(linewidth=np.inf, threshold=np.inf)
    setup_seed(args.seed)
    print(log_utils.get_option(args))

    # data and client
    datasetObj = data_utils.myDataset(args)
    dataset_train, dataset_test, dict_users, img_num_per_cls = datasetObj.get_imbalanced_dataset(datasetObj.get_args())

    # server and clients model
    num_classes = args.num_classes
    g_backbone = build_promptmodel(num_classes=num_classes, edge_size=224, patch_size=16,
                          Prompt_Token_num=args.prompt_token, VPT_type="Deep", embed_dim=args.embed_dim)
    g_backbone = g_backbone.to(args.device)
    g_classifier = nn.Linear(args.embed_dim, num_classes).to(args.device)

    client_num = args.num_users
    m = max(int(args.frac * client_num), 1)  # num_select_clients
    prob = [1 / client_num for i in range(client_num)]

    g_heads = [nn.Linear(args.embed_dim, num_classes) for i in range(args.num_users)]
    for i in range(args.num_users):
        g_heads[i] = g_heads[i].to(args.device)

    GPAs = [gpa.Gradient_adaptive_Prompt_Adjuster(args) for i in range(args.num_users)]

    # def count_parameters(model):
    #     total_params = sum(p.numel() for p in model.parameters())
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return total_params, trainable_params

    # total_params, trainable_params = count_parameters(g_backbone)
    # print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")

    # Training
    acc_list = []
    tail_acc_list = []
    for rnd in tqdm(range(args.rounds)):
        prompt_locals, head_w_locals, loss_locals = [], [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in range(args.num_users):
            local = update.LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            prompt_local, head_w_local = local.update_prompt(
                net=copy.deepcopy(g_backbone).to(args.device), head=g_heads[idx], GPA=GPAs[idx], epoch=args.local_ep)
            prompt_locals.append(copy.deepcopy(prompt_local))
            head_w_locals.append(copy.deepcopy(head_w_local))

        dict_len = [len(dict_users[idx]) for idx in idxs_users]

        # Aggregation
        prompt_avg, head_w_avg = fedavg.FedAvg_div(prompt_locals, head_w_locals, dict_len)
        print("round: ", rnd)
        print('prompt_avg: ', prompt_avg)
        g_backbone.load_prompt(prompt_avg)
        g_classifier.load_state_dict(copy.deepcopy(head_w_avg))
        acc, acc_per_class = update.globaltest(backbone=copy.deepcopy(g_backbone).to(args.device),
                                              classifier=copy.deepcopy(g_classifier).to(args.device),
                                              test_dataset=dataset_test, args=args)

        tail_acc = update.acc_for_segments(img_num_per_cls, acc_per_class)
        g_heads = [copy.deepcopy(g_classifier) for i in range(args.num_users)]

        acc_list.append(acc)
        tail_acc_list.append(tail_acc)

        print('round %d, global test acc %.4f, tail acc %.4f  \n' % (rnd, acc, tail_acc))
        print('-' * 30)

        print("savemodel...")
        torch.save(prompt_avg, os.path.join(output_folder, f'prompt_avg.pth'))
        torch.save(head_w_avg, os.path.join(output_folder, f'head_w_avg.pth'))

    print('acc_list', acc_list)
    print('tail_acc_list', tail_acc_list)
    torch.cuda.empty_cache()
