import torch
from collections import OrderedDict

def count_params(net):
    layers_data = {}
    all_param_types = set()

    for name, layer in net.named_children():
        state = layer.state_dict()
        if not state:
            continue
            
        layers_data[name] = {}
        for key, value in state.items():
            param_suffix = key.split(".")[-1]
            all_param_types.add(param_suffix)
            layers_data[name][param_suffix] = value.numel()

    sorted_types = sorted(list(all_param_types), reverse=True) 
    
    header = f"{'Layer Name':<15} | " + " | ".join([f"{t:<12}" for t in sorted_types]) + f" | {'Total':<10}"
    sep = "-" * len(header)
    
    print(header)
    print(sep)

    grand_total = 0
    for name, counts in layers_data.items():
        row_total = sum(counts.values())
        grand_total += row_total
        
        row_str = f"{name:<15} | "
        for p_type in sorted_types:
            val = counts.get(p_type, 0)
            row_str += f"{val:<12} | "
        
        row_str += f"{row_total:<10}"
        print(row_str)

    print(sep)
    print(f"{'TOTAL':<15} | " + " " * (len(header) - 30) + f" | {grand_total:<10}")