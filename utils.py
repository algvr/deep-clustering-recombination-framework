# imports needed for getattr_recursive

import os
import re
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms

eps = 1e-8


def safe_log(x):
    return torch.log(torch.max(torch.full_like(x, eps), x))


def safe_log_softmax(x, **kwargs):
    return F.log_softmax(torch.max(torch.full_like(x, eps), x), **kwargs)


def getattr_recursive(base_namespace, attr_name):
    # if attr_name contains at least one '.', consider attr to be "absolute" (torch.nn.functional.softmax)
    # rather than "relative" (functional.softmax)
    spl = attr_name.split('.')
    curr_level = base_namespace
    if '.' in attr_name:
        curr_level = globals()[spl[0]]
        spl.pop(0)
    for next_level in spl:
        curr_level = getattr(curr_level, next_level)
    return curr_level


def recursive_type_init(base_namespace, config, additional_keys_to_remove, callback=None):
    if isinstance(config, list):
        return list(map(lambda entry: recursive_type_init(base_namespace, entry, [], callback), config))
    elif isinstance(config, dict):
        class_name = config.get('type', None)
        args = {}
        positional_arguments = []
        for cfg_key in config:
            if cfg_key in additional_keys_to_remove or cfg_key == 'type':
                continue
            if cfg_key == 'positional_arguments':
                positional_arguments = recursive_type_init(base_namespace, config[cfg_key], [], callback)
            else:
                args[cfg_key] = recursive_type_init(base_namespace, config[cfg_key], [], callback)

        if class_name is not None:
            constructor_function = getattr_recursive(base_namespace, class_name)
            if callback is not None:
                return callback(constructor_function, positional_arguments, args)
            else:
                return constructor_function(*positional_arguments, **args)
        else:
            return args
    else:
        return config


def sanitize_filename(filename, default_if_empty=''):
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', filename.replace(' ', '_'))
    return default_if_empty if sanitized == '' else sanitized


def resolve_relative_path(path, base_path):
    if not os.path.isabs(path):
        if base_path is None:
            base_path = './' if '__file__' not in globals() else os.path.dirname(globals()['__file__'])
        path = os.path.abspath(os.path.join(base_path, path))
    if os.path.isdir(path) and not path.endswith(os.path.sep):
        return path + os.path.sep
    return path


def flatten(array):
    out = []
    for elem in array:
        if isinstance(elem, list):
            out.extend(flatten(elem))
        else:
            out.append(elem)
    return out


def get_cost_matrix(one_hot_assignments, one_hot_labels):
    num_clusters = one_hot_labels.shape[1]
    # uses negative of amount of correct assignments of samples to clusters under given cluster assignment as cost
    return -torch.matmul(one_hot_labels.transpose(0, 1), one_hot_assignments)


def get_performance_string(epoch_num, epoch_iter_num, last_performance_evaluation_acc,
                           last_performance_evaluation_nmi, last_performance_evaluation_ari, suffix=''):
    return 'Performance on epoch %i, iteration %i%s: ACC %.3f, NMI %.3f, ARI %.3f' % \
           (epoch_num, epoch_iter_num, ' ' + suffix if suffix != '' and suffix[0] != ' ' else suffix,
            last_performance_evaluation_acc, last_performance_evaluation_nmi, last_performance_evaluation_ari)


def evaluate_clustering_performance(one_hot_assignments, one_hot_labels):
    cost_mat = get_cost_matrix(one_hot_assignments, one_hot_labels).cpu().numpy()
    _, col_ind = linear_sum_assignment(cost_mat)
    # permute one_hot_labels according to mapping resulting from linear_sum_assignment
    perm_mat = F.one_hot(torch.from_numpy(col_ind).to(one_hot_labels.device)).float()  # num_classes not needed here
    one_hot_labels_perm = torch.matmul(one_hot_labels, perm_mat)
    one_hot_labels_perm_np = torch.argmax(one_hot_labels_perm, dim=1).cpu().numpy()
    one_hot_assignments_np = torch.argmax(one_hot_assignments, dim=1).cpu().numpy()
    acc = accuracy_score(one_hot_labels_perm_np, one_hot_assignments_np)
    nmi = normalized_mutual_info_score(one_hot_labels_perm_np, one_hot_assignments_np)
    ari = adjusted_rand_score(one_hot_labels_perm_np, one_hot_assignments_np)
    return acc, nmi, ari
