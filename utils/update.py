import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args

        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def get_loss(self):
        if self.args.loss_type == 'CE':
            return nn.CrossEntropyLoss()

    def train_test(self, dataset, idxs):
        train_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test_loader = DataLoader(dataset, batch_size=128)
        return train_loader, test_loader

    def update_prompt(self, net, head, GPA, epoch):
        net.train()
        head.train()
        backbone_optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        head_optimizer = torch.optim.SGD(head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for iter in range(epoch):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.long()
                backbone_optimizer.zero_grad()
                head_optimizer.zero_grad()
                feat = net(images)
                logits = head(feat)
                gpa = GPA(logits, labels)
                gpa.backward()
                # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=5.0)
                # torch.nn.utils.clip_grad_value_(GBA_Layer.parameters(), clip_value=5.0)
                backbone_optimizer.step()
                head_optimizer.step()
        return net.obtain_prompt_only(), head.state_dict()


def globaltest(backbone, test_dataset, args, classifier=None):
    backbone.eval()
    if classifier is not None:
        classifier.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    num_class = args.num_classes
    distri_class_label = [0 for i in range(num_class)]
    distri_class_correct = [0 for i in range(num_class)]

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            if classifier is not None:
                feat = backbone(images)
                outputs = classifier(feat)
            else:
                outputs = backbone(images, latent_output=False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                distri_class_label[int(label)] += 1
                if predicted[i] == label:
                    distri_class_correct[label] += 1

    acc_per_class = [distri_class_correct[i] / distri_class_label[i] for i in range(num_class)]
    print("acc_per_class:", acc_per_class)
    acc = correct / total
    return acc, acc_per_class


def acc_for_segments(img_num_per_cls, acc_per_class):
    n = len(img_num_per_cls)

    tail_threshold = n // 5
    mid_upper_threshold = n * 3 // 5

    sorted_indices = sorted(range(n), key=lambda i: img_num_per_cls[i])

    tail_indices = sorted_indices[:tail_threshold]
    mid_indices = sorted_indices[tail_threshold:mid_upper_threshold]
    head_indices = sorted_indices[mid_upper_threshold:]

    rare_indices = [i for i, x in enumerate(img_num_per_cls) if x <= 10]

    tail_values = [acc_per_class[i] for i in tail_indices]
    mid_values = [acc_per_class[i] for i in mid_indices]
    head_values = [acc_per_class[i] for i in head_indices]

    rare_values = [acc_per_class[i] for i in rare_indices]

    tail_avg = sum(tail_values) / len(tail_values) if tail_values else 0
    mid_avg = sum(mid_values) / len(mid_values) if mid_values else 0
    head_avg = sum(head_values) / len(head_values) if head_values else 0
    rare_avg = sum(rare_values) / len(rare_values) if rare_values else 0

    return rare_avg
