import argparse
import random
import numpy as np
import torch
import re

import yaml


class Namespace(object):
    """
    load the configs from xxxx.yaml
    args.key
    """

    def __init__(self, dic):
        for key, value in dic.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in config file!")


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args(config_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=config_dir, help="config film, xxxx.yaml")

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    # if args.debug:
    #     if args.train:
    #         args.train.batch_size = 4
    #         args.train.epochs = 10
    #     if args.eval:
    #         args.eval.batch_size = 2
    #         # retrain 1 epoch
    #         args.eval.epochs = 1
    #     args.dataset.num_workers = 0

    # vars(args)['aug_kwargs'] = {
    #     'name': args.model.name,
    #     'image_size': args.dataset.image_size,
    #     'dataset': args.dataset.name
    # }
    # vars(args)['dataset_kwargs'] = {
    #     'name': args.dataset.name,
    #     'data_dir': args.dataset.data_dir,
    #     'download': args.dataset.download,
    #     'debug_subset_size': args.subset_size if args.debug else None,
    # }
    # vars(args)['dataloader_kwargs'] = {
    #     'drop_last': True,
    #     'pin_memory': True,
    #     'num_workers': args.dataset.num_workers
    # }

    return args

