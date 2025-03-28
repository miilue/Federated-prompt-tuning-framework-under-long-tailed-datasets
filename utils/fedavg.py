import copy
import torch


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = w_avg[k] / len(w)
    return w_avg


def FedAvg_noniid(w, dict_len):
    with torch.no_grad():
        w_avg = copy.deepcopy(w[0])
        if isinstance(w_avg, torch.nn.Parameter):
            w_avg = w_avg * dict_len[0]
            for i in range(1, len(w)):
                w_avg += w[i] * dict_len[i]
            w_avg = w_avg / sum(dict_len)
        else:
            for k in w_avg.keys():
                w_avg[k] = w_avg[k] * dict_len[0]
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k] * dict_len[i]
                w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg


def FedAvg_div(prompt_locals, head_w_locals, dict_len):
    prompt_avg = FedAvg_noniid(prompt_locals, dict_len)
    head_w_avg = FedAvg_noniid(head_w_locals, dict_len)
    return prompt_avg, head_w_avg


def FedAvg_Vit(prompt_head_locals, dict_len):
    linear_w_avg = FedAvg_noniid([local['head'] for local in prompt_head_locals], dict_len)
    prompt_avg = FedAvg_noniid([local['Prompt_Tokens'] for local in prompt_head_locals], dict_len)
    return {'head': linear_w_avg, 'Prompt_Tokens': prompt_avg}
