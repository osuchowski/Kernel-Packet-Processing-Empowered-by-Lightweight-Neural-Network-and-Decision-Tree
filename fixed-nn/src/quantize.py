"""
Script for PTQ using pytorch-quantization package
"""
import argparse
from pathlib import Path
import os
import sys
import pickle
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neural_nets import MLP, ConvNet
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch.utils.data import DataLoader, random_split
from train_mlp import PacketDataset, PacketFlowDataset, get_dataset, get_nth_split
torch.multiprocessing.set_sharing_strategy('file_system')


def collect_stats(model, data_loader, num_bins):
    """Feed data to the network and collect statistic"""
    is_mlp = isinstance(model, MLP)
    model.eval()
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
                if isinstance(module._calibrator, calib.HistogramCalibrator):
                    module._calibrator._num_bins = num_bins
            else:
                module.disable()

    for batch, _ in data_loader:
        if is_mlp:
            x = batch.float()
        else:
            x = batch.unsqueeze(1).float()
        model(x)

        # Disable calibrators
        for _, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")


def quantize_model_params(model):
    """Quantize layer weights using calculated amax
       and process scale constant for C-code

    Args:
        state_dict (Dict): pytorch model state_dict
        amax (Dict): dictionary containing amax values
    """

    is_mlp = isinstance(model, MLP)

    indices = [0, 2, 4] if is_mlp else [0, 3, 7] 
    # indices = [0, 2] if is_mlp else [0, 3, 7] 
    scale_factor = 127 # 127 for 8 bits
    # scale_factor = 2**(16-1) - 1# 127 for 8 bits

    state_dict = dict()

    for layer_idx, idx in enumerate(indices, start=1):
        # quantize all parameters
        weight = model.state_dict()[f'net.{idx}.weight']
        s_w = model.state_dict()[f'net.{idx}._weight_quantizer._amax'].numpy()
        s_x = model.state_dict()[f'net.{idx}._input_quantizer._amax'].numpy()

        scale = weight * (scale_factor / s_w)
        state_dict[f'layer_{layer_idx}_weight'] = torch.clamp(scale.round(), min=-scale_factor, max=scale_factor).to(int)
        state_dict[f'layer_{layer_idx}_weight'] = torch.clamp(scale.round(), min=-scale_factor, max=scale_factor).to(int)
        if is_mlp or layer_idx == 3:
            state_dict[f'layer_{layer_idx}_weight'] = state_dict[f'layer_{layer_idx}_weight'].T
        state_dict[f'layer_{layer_idx}_weight'] = state_dict[f'layer_{layer_idx}_weight'].numpy()

        state_dict[f'layer_{layer_idx}_s_x'] = scale_factor / s_x
        state_dict[f'layer_{layer_idx}_s_x_inv'] = s_x / scale_factor
        state_dict[f'layer_{layer_idx}_s_w_inv'] = (s_w / scale_factor).squeeze()
        # print(layer_idx, idx, weight, scale, s_x, s_w, scale_factor / s_x, s_x / scale_factor, (s_w / scale_factor).squeeze())

    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='mlp_pktflw.th')
    parser.add_argument('--num_bins', help='number of bins', type=int, default=128)
    parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='../dataset')
    parser.add_argument('--save_dir', help='save directory', default='../saved_models', type=Path)
    parser.add_argument('--maxLength', type=int, default=100, help='max length')
    parser.add_argument('--fold', type=int, default=0, help='fold to use')
    parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
    parser.add_argument('--maxSize', type=int, default=sys.maxsize, help='limit of samples to consider')
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.8)
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--calib-method', default='histogram', type=str, help='calibration method')
    parser.add_argument('--is-binary', action='store_true')

    args = parser.parse_args()

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # load model
    saved_stats = torch.load(os.path.join(args.save_dir, args.filename), weights_only=False)

    calib_method = args.calib_method

    state_dict = saved_stats['state_dict']
    scaler = saved_stats.get('scaler')

    hidden_sizes = None if 'convnet' in args.filename else saved_stats['hidden_sizes']
    channel_sizes = None if 'mlp' in args.filename else saved_stats['channel_sizes']


    quant_nn.QuantLinear.set_default_quant_desc_input(QuantDescriptor(num_bits=8, calib_method=calib_method))
    quant_nn.QuantLinear.set_default_quant_desc_input(QuantDescriptor(num_bits=8, calib_method=calib_method))
    # quant_nn.QuantLinear.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram'))
    # quant_nn.QuantConv2d.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram'))
    quant_modules.initialize()


    in_dim = 12
    if args.is_binary:
        out_dim = 2
    else:
        out_dim = 7
    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden_sizes=hidden_sizes) if 'mlp' in args.filename else ConvNet(in_dim=in_dim, channel_sizes=channel_sizes, out_dim=out_dim)
    model.load_state_dict(state_dict)
    #
    with open (os.path.join(args.data_dir, 'flows.pickle'), "rb") as f:
        all_data = pickle.load(f)

    with open(os.path.join(args.data_dir, "flows_categories_mapping.json"), "r") as f:
        categories_mapping_content = json.load(f)

    categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
    assert min(mapping.values()) == 0

    all_data = [item[:args.maxLength,:] for item in all_data if np.all(item[:,4]>=0)]
    random.shuffle(all_data)
    # print("lens", [len(item) for item in all_data])
    x = [item[:, :-2] for item in all_data]
    y = [item[:, -1:] for item in all_data]
    categories = [item[:, -2:-1] for item in all_data]

    dataset = PacketDataset(x, y, categories)

    # split training data to train/validation
    split_r = args.train_val_split

    # globals()[args.function]()
    n_fold = args.nFold
    fold = args.fold

    train_indices, test_indices = get_nth_split(dataset, n_fold, fold)
    train_data = torch.utils.data.Subset(dataset, train_indices)
    # test_data = torch.utils.data.Subset(dataset, test_indices)

    new_dataset_train, new_labels_train = get_dataset(train_data, args.is_binary, scaler=scaler)
    # new_dataset_test, new_labels_test = get_dataset(test_data)
    train_trainset = PacketFlowDataset(new_dataset_train, new_labels_train)
    # test_testset = PacketFlowDataset(new_dataset_test, new_labels_test)
    train_trainset, train_valset = random_split(train_trainset, [round(len(train_trainset)*split_r), round(len(train_trainset)*(1 - split_r))], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_trainset, batch_size=len(train_trainset), num_workers=1, shuffle=True)
    print(len(train_trainset))

    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, train_loader, args.num_bins)
        compute_amax(model, method="entropy")
        # compute_amax(model, method="percentile")
        # compute_amax(model, method="mse")

    state_dict = quantize_model_params(model)
    saved_stats['state_dict'] = state_dict

    name = args.filename.replace('.th', '_quant.th')

    torch.save(saved_stats, os.path.join(args.save_dir, name))

    name = args.filename.replace('.th', f'_quant_{calib_method}.th')
    torch.save(saved_stats, os.path.join(args.save_dir, name))
