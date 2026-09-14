"""
Script for writing param header and source files in C with weights and amax values calculate in python
"""
import argparse
from pathlib import Path
import os
import numpy as np
import json

import torch

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_output_dim(input_dim, kernel_size, stride):
            output_dim = (input_dim -(kernel_size-1) - 1) / stride
            return int(output_dim + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='mlp_pktflw_quant.th')
    parser.add_argument('--bpf-text', help='filename', action='store_true')
    parser.add_argument('--binary', help='filename', action='store_true')
    parser.add_argument('--save_dir', help='save directory', default='../saved_models', type=Path)

    args = parser.parse_args()

    saved_stats = torch.load(os.path.join(args.save_dir, args.filename), weights_only=False)
    state_dict = saved_stats['state_dict']
    hidden_sizes = saved_stats['hidden_sizes']

    ret = {}

    scaler = saved_stats.get('scaler')
    data_min_fxp = None
    data_scale_fxp = None
    if scaler is not None:
        data_min_fxp = (scaler.data_min_ * (2 ** 16)).round().astype(np.int64).tolist()
        data_scale_fxp = (scaler.scale_ * (2 ** 16)).round().astype(np.int64).tolist()
        ret["data_min"] = data_min_fxp
        ret["data_scale"] = data_scale_fxp
    else:
        print("Warning: 'scaler' not found in saved checkpoint. data_min and data_scale will not be exported.")
    # create header file
    with open('../src/mlp_params.h', 'w') as f:
        f.write('/*******************************************************************\n')
        f.write('@file mlp_params.h\n*  @brief variable prototypes for model parameters and amax values\n*\n*\n')
        f.write('*  @author Benjamin Fuhrer\n*\n')
        f.write('*******************************************************************/\n')
        f.write('#ifndef MLP_PARAMS\n#define MLP_PARAMS\n\n')

        f.write(f'#define INPUT_DIM {12}\n')
        for idx, hidden_size in enumerate(hidden_sizes, start=1):
            f.write(f'#define H{idx} {hidden_size}\n')

        f.write(f'//#define IS_BINARY\n')
        f.write(f'#ifdef IS_BINARY\n')
        f.write(f'#define OUTPUT_DIM {1}\n')
        f.write(f'#else\n')
        if args.binary:
            f.write(f'#define OUTPUT_DIM {2}\n')
        else:
            f.write(f'#define OUTPUT_DIM {7}\n')
        f.write(f'#endif\n')
        f.write('#include <stdint.h>\n\n\n')

        if data_min_fxp is not None:
            f.write('// normalization constants\n')
            f.write(f"extern const int64_t data_min[{len(data_min_fxp)}];\n")
            f.write(f"extern const int64_t data_scale[{len(data_scale_fxp)}];\n\n")


        f.write('// quantization/dequantization constants\n')

        for layer_idx in range(1, 4):

            name = f'layer_{layer_idx}_s_x'
            f.write(f"extern const int {name};\n")

            name = f'layer_{layer_idx}_s_x_inv'
            f.write(f"extern const int {name};\n")

            name = f'layer_{layer_idx}_s_w_inv'
            value = state_dict[name]
            f.write(f"extern const int {name}[{len(value)}];\n")

            # name = f'layer_{layer_idx}_s_x_f'
            # f.write(f"extern const float {name};\n")
            #
            # name = f'layer_{layer_idx}_s_x_inv_f'
            # f.write(f"extern const float {name};\n")
            #
            # name = f'layer_{layer_idx}_s_w_inv_f'
            # value = state_dict[name.replace('_f', '')]
            # f.write(f"extern const float {name}[{len(value)}];\n")
            #

        f.write('// Layer quantized parameters\n')
        for layer_idx in range(1, 4):
            name = f'layer_{layer_idx}_weight'
            param = state_dict[f'layer_{layer_idx}_weight']
            f.write(f"extern const int8_t {name}[{len(param.flatten())}];\n")

        f.write('\n#endif // end of MLP_PARAMS\n')

    # create source file
    with open('../src/mlp_params.c', 'w') as f:
        if args.bpf_text:
            pass
        else:
            f.write('#include "mlp_params.h"\n\n\n')
            if data_min_fxp is not None:
                f.write(f"const int64_t data_min[{len(data_min_fxp)}] = {{")
                f.write(", ".join(map(str, data_min_fxp)))
                f.write("};\n\n")

                f.write(f"const int64_t data_scale[{len(data_scale_fxp)}] = {{")
                f.write(", ".join(map(str, data_scale_fxp)))
                f.write("};\n\n")

        for layer_idx in range(1, 4):
            name = f'layer_{layer_idx}_s_x'
            fxp_value = (state_dict[name] * (2**16)).round()
            f.write(f"const int {name} = {int(fxp_value)};\n\n")
            ret[name] = []
            ret[name].append(int(fxp_value))

            name = f'layer_{layer_idx}_s_x_inv'
            fxp_value = (state_dict[name] * (2**16)).round()
            f.write(f"const int {name} = {int(fxp_value)};\n\n")
            ret[name] = []
            ret[name].append(int(fxp_value))

            name = f'layer_{layer_idx}_s_w_inv'
            name2 = f'layer_{layer_idx}_s_x_inv'
            fxp_value = (((state_dict[name] * (2**16)).round() * (state_dict[name2] * (2**16)).round()) / (2**16)).round()
            ret[name] = []

            for idx in range(len(fxp_value)):
                ret[name].append(int(fxp_value[idx]))

            name = f'layer_{layer_idx}_s_w_inv'
            fxp_value = (state_dict[name] * (2**16)).round()
            f.write(f"const int {name}[{len(fxp_value)}] = {{")

            for idx in range(len(fxp_value)):
                f.write(f"{int(fxp_value[idx])}")
                if idx < len(fxp_value) - 1:
                     f.write(", ")
            f.write("};\n\n")

            # name = f'layer_{layer_idx}_s_x'
            # value = state_dict[name]
            # f.write(f"const float {name}_f = {float(value)};\n\n")
            #
            # name = f'layer_{layer_idx}_s_x_inv'
            # value = state_dict[name]
            # f.write(f"const float {name}_f = {float(value)};\n\n")
            #
            # name = f'layer_{layer_idx}_s_w_inv'
            # value = state_dict[name]
            # f.write(f"const float {name}_f[{len(value)}] = {{")
            #
            # for idx in range(len(value)):
            #     f.write(f"{float(value[idx])}")
            #     if idx < len(value) - 1:
            #          f.write(", ")
            # f.write("};\n\n")


        for layer_idx in range(1, 4):
                name = f'layer_{layer_idx}_weight'
                ret[name] = []
                param = state_dict[f'layer_{layer_idx}_weight']
                param = param.flatten()
                f.write(f"const int8_t {name}[{len(param)}] = {{")
                for idx in range(len(param)):
                    f.write(f"{param[idx]}")
                    ret[name].append(param[idx])
                    if idx < len(param) - 1:
                        f.write(", ")
                f.write("};\n")
    with open("../src/mlp_params.json", "w") as f:
        json.dump(ret, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '), cls=NpEncoder)
