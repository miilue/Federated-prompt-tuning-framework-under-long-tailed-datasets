from http import client
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch.utils
from torch import Tensor
from torch.utils.data import Dataset, Subset
from collections import Counter
import resource
import math


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)]  # initial user and index for whole dataset
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))  # 'replace=False' make sure that there is no repeat
        all_idxs = list(set(all_idxs)-dict_users[i])
    return dict_users


def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    np.random.seed(seed)
    p = 1
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(Phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client==0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j]==1)[0]) for j in range(num_classes)]   # indicate the clients that choose each class
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}
    for class_i in range(num_classes):
        all_idxs = np.where(y_train==class_i)[0]

        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])
    return dict_users


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(self.img_num_per_cls)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        print('img_num_per_cls: ', img_num_per_cls)
        print('sum_img_num_per_cls: ', sum(img_num_per_cls))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class ImbalancedData(Dataset):
    def __init__(self, dataset_name, root, train=True, imb_type='exp', imb_factor=0.01, rand_number=0, transform=None, download=False):
        self.transform = transform
        if not train:
            if dataset_name == 'oxford_pets':
                self.dataset = datasets.OxfordIIITPet(root=root, split='test', download=download)
            elif dataset_name == 'dtd':
                self.dataset = datasets.DTD(root=root, split='test', download=download, transform=None,
                                            target_transform=None)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            try:
                self.targets = self.dataset.labels
            except AttributeError:
                try:
                    self.targets = self.dataset.targets
                except AttributeError:
                    self.targets = [sample[1] for sample in self.dataset]
        else:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"Soft limit: {soft}, Hard limit: {hard}")
            np.random.seed(rand_number)
            if dataset_name == 'oxford_pets':
                raw_dataset = datasets.OxfordIIITPet(root=root, split='trainval', download=download)
            elif dataset_name == 'dtd':
                train_dataset = datasets.DTD(root=root, split='train', download=download, transform=None,
                                            target_transform=None)
                val_dataset = datasets.DTD(root=root, split='val', download=download, transform=None,
                                           target_transform=None)
                raw_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            try:
                self.labels = raw_dataset.labels
            except AttributeError:
                try:
                    self.labels = raw_dataset.targets
                except AttributeError:
                    self.labels = [sample[1] for sample in raw_dataset]

            self.labels_np = np.array(self.labels, dtype=np.int64)
            self.classes = np.unique(self.labels)
            self.cls_num = len(self.classes)
            self.dataset = self.gen_imbalanced_data(raw_dataset, imb_type, imb_factor)
            self.targets = [sample[1] for sample in self.dataset]
            final_img_num_per_cls = self.get_cls_num_list()
            self.img_num_per_cls = final_img_num_per_cls
            print('final_img_num_per_cls: ', final_img_num_per_cls)
            print('sum_img_num_per_cls: ', sum(final_img_num_per_cls))

    def get_img_num_per_cls(self, label_count, imb_type, imb_factor):
        _, img_max = label_count[0]
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(self.cls_num):
                num = max(img_max * (imb_factor ** (cls_idx / (self.cls_num - 1.0))), 1)
                if self.cls_num == 47:
                    img_num_per_cls.append(math.ceil(num))
                else:
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * self.cls_num)

        print('img_num_per_cls: ', img_num_per_cls)
        return img_num_per_cls

    def gen_imbalanced_data(self, dataset, imb_type, imb_factor):
        if imb_factor == 1.0:
            return dataset

        label_count = Counter(self.labels).most_common(self.cls_num)
        print('label_count: ', label_count)
        img_num_per_cls = self.get_img_num_per_cls(label_count, imb_type, imb_factor)
        img_num_per_cls_reassigned = [0] * self.cls_num
        for i in range(self.cls_num):
            label_idx, _ = label_count[i]
            img_num_per_cls_reassigned[label_idx] = img_num_per_cls[i]
        print('img_num_per_cls_reassigned: ', img_num_per_cls_reassigned)

        num_per_cls_dict = dict()
        select_idx = []
        for the_class, the_img_num in zip(self.classes, img_num_per_cls_reassigned):
            num_per_cls_dict[the_class] = the_img_num
            idx = np.where(self.labels_np == the_class)[0]
            np.random.shuffle(idx)
            select_idx.extend(idx[:the_img_num])

        imbalanced_dataset = Subset(dataset, select_idx)
        return imbalanced_dataset

    def get_cls_num_list(self):
        cls_num_list = [0] * len(np.unique(self.targets))
        for label in self.targets:
            cls_num_list[label] += 1
        return cls_num_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self.dataset, '__getitem__'):
            item = self.dataset[idx]
            if isinstance(item, tuple) and len(item) == 2:
                img, label = item
            else:
                img = item
                label = self.targets[idx]
        else:
            img = self.dataset.data[idx]
            label = self.targets[idx]
        if self.transform:
            if len(img.getbands()) == 1:
                img = transforms.Grayscale(num_output_channels=3)(img)
            img = self.transform(img)
        return img, label


class myDataset():
    def __init__(self, args):
        self.m_args = args
        self.dataset_name = {'oxford_pets', 'dtd'}

    def get_args(self):
        return self.m_args

    def get_imbalanced_dataset(self, args):
        args.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        print(args.device)

        if args.dataset == 'cifar10':
            data_path = './data/cifar_lt/'
            args.num_classes = 10
            trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize([224, 224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            trans_val = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            dataset_train = IMBALANCECIFAR10(data_path, imb_factor=args.IF, train=True, download=True,
                                             transform=trans_train)
            dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)

        elif args.dataset == 'cifar100':
            data_path = './data/cifar_lt/'
            args.num_classes = 100
            trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize([224, 224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            trans_val = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            dataset_train = IMBALANCECIFAR100(data_path, imb_factor=args.IF, train=True, download=True,
                                             transform=trans_train)
            dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)

        elif args.dataset in self.dataset_name:
            if args.dataset == 'oxford_pets':
                args.varphi = 6.0
            data_path = os.path.join("data", args.dataset + '_lt')
            trans_train = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            trans_val = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            dataset_train = ImbalancedData(args.dataset, data_path, train=True, imb_factor=args.IF, download=True,
                                             transform=trans_train)
            args.num_classes = dataset_train.cls_num
            print('num_classes: ', args.num_classes)
            dataset_test = ImbalancedData(args.dataset, data_path, train=False, download=True, transform=trans_val)
        else:
            exit('Error: unrecognized dataset')

        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
        n_test = len(dataset_test)
        y_test = np.array(dataset_test.targets)
        img_num_per_cls = dataset_train.img_num_per_cls

        if args.iid:
            print("Into iid sampling")
            dict_users = iid_sampling(n_train, args.num_users, args.seed)

        else:
            print("Into non-iid sampling")
            dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users,
                                                    args.seed, args.alpha_dirichlet)
        clients_sizes = [len(dict_users[i]) for i in range(args.num_users)]
        print("clients_sizes: {}".format(clients_sizes))
        map_testset = {}
        for i in range(args.num_classes):
            idxs = []
            for j in range(min(100 * args.num_classes, len(y_test))):
                if y_test[j] == i:
                    idxs.append(j)
            map_testset[i] = idxs
        assert len(map_testset) == args.num_classes

        alist = np.array(
            [[np.sum(y_train[list(dict_users[i])] == j) for j in range(args.num_classes)] for i in range(len(clients_sizes))])
        print("training set distribution:")
        print(alist)
        print("Total size of training set")
        print(sum(alist.sum(0)))
        self.training_set_distribution = alist  # training_set_distribution[i, j]: sample size of clien i and class j
        return dataset_train, dataset_test, dict_users, img_num_per_cls
