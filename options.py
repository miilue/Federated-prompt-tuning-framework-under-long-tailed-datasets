import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=30, help="number of communication rounds")
    parser.add_argument('--local_ep', type=int, default=2, help="number of local epochs")
    parser.add_argument('--frac', type=float, default=1, help="fration of selected clients")
    parser.add_argument('--num_users', type=int, default=10, help="number of uses")
    parser.add_argument('--local_bs', type=int, default=12, help="local batch size")
    parser.add_argument('--cropsize', type=int, default=224, help="data_cropsize")
    parser.add_argument('--embed_dim', type=int, default=768, help="embed_dim")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--model', type=str, default='ViT', help="model name")
    parser.add_argument('--prompt_token', type=int, default=10, help="number of prompt tokens")
    parser.add_argument('--dataset', type=str, default='cifar100', help="dataset, option: cifar100, oxford_pets, dtd")
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float, default=1, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=0.5, help="Dirichlet distribution")
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
    parser.add_argument('--IF', type=float, default=0.01, help="imbalance factor: Min/Max")
    parser.add_argument('--varphi', type=float, default=4.0, help="weight coefficient varphi")
    parser.add_argument('--gpu', type=int, default=0, help="gpu")
    return parser.parse_args()
