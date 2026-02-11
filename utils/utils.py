import torch
from torch import nn
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchinfo import summary

def count_params(net):
    layers_data = {}
    all_param_types = set()

    for name, layer in net.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer_id = f"{name} ({layer.__class__.__name__})"
            layers_data[layer_id] = {}
            
            for param_name, param in layer.named_parameters():
                param_suffix = param_name.split(".")[-1] 
                all_param_types.add(param_suffix)
                layers_data[layer_id][param_suffix] = param.numel()

    if not layers_data:
        print("No layers with parameters found.")
        return

    sorted_types = sorted(list(all_param_types), reverse=True) 
    header = f"{'Layer Name':<40} | " + " | ".join([f"{t:<12}" for t in sorted_types]) + f" | {'Total':<10}"
    sep = "-" * len(header)

    print(header)
    print(sep)

    grand_total = 0
    for name, counts in layers_data.items():
        row_total = sum(counts.values())
        grand_total += row_total
        
        row_str = f"{name:<40} | "
        for p_type in sorted_types:
            val = counts.get(p_type, 0)
            row_str += f"{val:<12} | "
        
        row_str += f"{row_total:<10}"
        print(row_str)

    print(sep)
    print(f"{'TOTAL':<40} | " + " " * (len(header) - 55) + f" | {grand_total:,}")

    return grand_total

def compute_FLOPS(model, input_size):
    curr_size = input_size
    total_flops = 0

    for m in model.modules():
        if len(list(m.children())) > 0:
            continue

        if isinstance(m, nn.Conv2d):
            out_size = ((curr_size - m.kernel_size[0] + 2 * m.padding[0]) // m.stride[0]) + 1
            
            layer_flops = 2 * (m.out_channels * out_size**2) * \
                             (m.in_channels * m.kernel_size[0] * m.kernel_size[1])
            
            total_flops += layer_flops
            curr_size = out_size 
            print(f"Conv: {m.in_channels}->{m.out_channels}, Size: {curr_size}, FLOPs: {layer_flops:.0f}")

        elif isinstance(m, nn.MaxPool2d):
            padding = m.padding if isinstance(m.padding, int) else m.padding[0]
            stride = m.stride if isinstance(m.stride, int) else m.stride[0]
            kernel = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            
            curr_size = ((curr_size - kernel + 2 * padding) // stride) + 1
            print(f"Maxpool: New Size {curr_size}")

        elif isinstance(m, nn.Linear):
            layer_flops = 2 * m.in_features * m.out_features
            total_flops += layer_flops
            print(f"Linear: FLOPs: {layer_flops:.0f}")

        elif isinstance(m, nn.AdaptiveAvgPool2d):
            curr_size = m.output_size[0] if isinstance(m.output_size, tuple) else m.output_size
            print(f"AvgPool: New Size {curr_size}")

    return total_flops


def get_optimizer(optimizer_name: str, model_parameters, lr=0.001, **kwargs):
    params = list(model_parameters)
    if not params:
        raise ValueError("Model parameters are empty.")

    name = optimizer_name.lower()
    
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, **kwargs)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, **kwargs)
    else:
        return torch.optim.Adam(params, lr=lr, **kwargs)
    
def get_loss_function(loss_name: str):
    losses = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "L1": nn.L1Loss()
    }

    return losses.get(loss_name.lower(), nn.CrossEntropyLoss())

def get_gpu():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def set_random_seeds(seed):

    device = get_gpu()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if device == "mps":
        torch.mps.manual_seed(seed)

def min_max_scaling(image):
    min = image.min()
    max = image.max()
    return (image - min) / (max - min + 1e-5)

def plot_images(images, labels, classes, normalize=False, probs=None, true_labels=None):
    n_images = len(images)

    n_cols = int(np.sqrt(n_images))
    n_rows = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(10, 10))

    for i in range(n_rows*n_cols):
        ax = fig.add_subplot(n_rows, n_cols, i+1)

        image = images[i]
        image_class = classes[labels[i]]

        image = image.permute(1, 2, 0).cpu().numpy()

        if normalize:
            image = min_max_scaling(image)

        ax.imshow(image)
        ax.set_title(image_class)
        ax.axis("off")

    plt.tight_layout()

def plot_convolutions(filter, images, labels, classes, normalize=False):

    filter = torch.FloatTensor(filter)
    filter = filter.repeat(3, 3, 1, 1)

    images = [*images][:5]
    images = torch.stack(images)

    feature_maps = F.conv2d(images, filter)

    n_images = len(images)
    n_rows = 2
    n_cols = n_images

    images = images.permute(0, 2, 3, 1)
    feature_maps = feature_maps.permute(0, 2, 3, 1)

    fig = plt.figure(figsize=(10, 10))

    for i in range(n_images):

        ax1 = fig.add_subplot(n_rows, n_cols, i+1)
        ax2 = fig.add_subplot(n_rows, n_cols, i+n_images+1)

        if normalize:
            image = min_max_scaling(images[i])
            feature_map = min_max_scaling(feature_maps[i])

        ax1.imshow(image)
        ax2.imshow(feature_map)

        image_name = classes[labels[i]]
        ax1.set_title(f"Original: {image_name}")
        ax2.set_title(f"Feature map: {image_name}")

        ax1.axis("off")
        ax2.axis("off")

    fig.tight_layout()

def plot_subsample(kernel_size, mode:str, images, labels, classes, normalize=False):

    images = [*images][:5]
    images = torch.stack(images)

    if mode == "avg":
        subsampled = F.avg_pool2d(input=images, kernel_size=kernel_size)
    elif mode == "max":
        subsampled = F.max_pool2d(input=images, kernel_size=kernel_size)
    else: 
        return

    n_images = len(images)
    n_rows = 2
    n_cols = n_images

    images = images.permute(0, 2, 3, 1)
    subsampled = subsampled.permute(0, 2, 3, 1)

    fig = plt.figure(figsize=(10, 10))

    for i in range(n_images):

        ax1 = fig.add_subplot(n_rows, n_cols, i+1)
        ax2 = fig.add_subplot(n_rows, n_cols, i+n_images+1)

        if normalize:
            image = min_max_scaling(images[i])
            feature_map = min_max_scaling(subsampled[i])

        ax1.imshow(image)
        ax2.imshow(feature_map)

        image_name = classes[labels[i]]
        ax1.set_title(f"Original: {image_name}")
        ax2.set_title(f"Subsampled: {image_name}")

        ax1.axis("off")
        ax2.axis("off")

    fig.tight_layout()

def get_top_k_confident_mistakes(y_hat, y_true, probs, k=5):
    
    incorrect_mask = (y_hat != y_true)
    
    incorrect_indices = torch.nonzero(incorrect_mask).flatten()
    
    if len(incorrect_indices) == 0:
        print("No mistakes found! Perfect accuracy?")
        return None

    mistake_probs = probs[incorrect_mask]
    mistake_y_hat = y_hat[incorrect_mask]
    mistake_y_true = y_true[incorrect_mask]
    
   
    top_values, top_rel_indices = torch.topk(mistake_probs, k=min(k, len(mistake_probs)))
    
   
    final_indices = incorrect_indices[top_rel_indices]
    final_y_hat = mistake_y_hat[top_rel_indices]
    final_y_true = mistake_y_true[top_rel_indices]
    
    results = []
    for i in range(len(final_indices)):
        results.append({
            "dataset_index": final_indices[i].item(),
            "predicted_class": final_y_hat[i].item(),
            "true_class": final_y_true[i].item(),
            "confidence": top_values[i].item()
        })
        
    return results

def plot_top_mistakes(mistakes, classes, trainer):
    fig = plt.figure(figsize=(10, 10))
    n_rows = int(np.sqrt(len(mistakes)))
    n_cols = int(np.sqrt(len(mistakes)))

    for i, m in enumerate(mistakes):
        idx = m['dataset_index']
        img, _ = trainer.data.testloader.dataset[idx]

        img = min_max_scaling(img)

        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"Pred: {classes[m['predicted_class']]} | True: {classes[m['true_class']]}\nConf: {m['confidence']:.4f}")
        ax.axis("off")

    fig.tight_layout()

def plot_training_results(*histories, titles=None, figsize=None):
    num_plots = len(histories)
    if titles is None:
        titles = [f"Experiment {i+1}" for i in range(num_plots)]
    
    if figsize is None:
        figsize = (8 * num_plots, 6)
        
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, squeeze=False)

    for i, (history, title) in enumerate(zip(histories, titles)):
        ax = axes[0, i]
        epochs = range(1, len(history["train_loss"]) + 1)
        
        ax.plot(epochs, history["train_loss"], label='Train Loss', color='blue')
        ax.plot(epochs, history["val_loss"], label='Test Loss', color='orange')
        
        if "val_err" in history:
            ax.plot(epochs, history["val_err"], label='Test Error', color='red', linestyle='--')
            
            min_err = min(history["val_err"])
            ax.axhline(y=min_err, color='gray', linestyle='-.', alpha=0.5, 
                       label=f'Min Error: {min_err:.4f}')
            
            ax.text(1, min_err, f' Min: {min_err:.2%}', color='gray', 
                    va='bottom', ha='left', fontweight='bold')
            
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
        max_val = max(max(history["train_loss"]), 2.0)
        ax.set_ylim(0, max_val)

    plt.tight_layout()
    plt.show()

def plot_filters_and_feature_maps(images, model, n_images=5, n_filters=7, features=False):
    if features:
        filters = model.features[0].weight.data[:n_filters]
        bias = model.features[0].bias.data[:n_filters]
    else:
        filters = model.net[0].weight.data[:n_filters]
        bias = model.net[0].bias.data[:n_filters]

    images_ = images[:n_images].to(next(model.parameters()).device)
    
    with torch.no_grad():
        feature_maps = F.conv2d(input=images_, weight=filters, bias=bias, stride=4)

    fig, axes = plt.subplots(n_images + 1, n_filters + 1, figsize=(20, 3 * (n_images + 1)))

    axes[0, 0].axis('off') 
    axes[0, 0].text(0.5, 0.5, "FILTERS", ha='center', va='center', weight='bold')
    
    for j in range(n_filters):
        f = filters[j].detach().cpu()
        f = min_max_scaling(f).permute(1, 2, 0).numpy()
        axes[0, j+1].imshow(f, cmap="bone")
        axes[0, j+1].set_title(f"Filter {j}")
        axes[0, j+1].axis('off')

    for i in range(n_images):
        img = images_[i].detach().cpu()
        img = min_max_scaling(img).permute(1, 2, 0).numpy()
        axes[i+1, 0].imshow(img, cmap="bone")
        axes[i+1, 0].set_title(f"Input {i}", weight='bold')
        axes[i+1, 0].axis('off')
        
        for j in range(n_filters):
            f_map = feature_maps[i, j].detach().cpu().numpy()
            axes[i+1, j+1].imshow(f_map, cmap='bone') 
            axes[i+1, j+1].axis('off')

    plt.tight_layout()
    plt.show()

def compare_architectures(model_dict, input_size=(1, 3, 224, 224), colors=None):
    model_names = list(model_dict.keys())
    if colors is None:
        colors = ['#1abc9c', '#3498db', '#e74c3c', '#9b59b6', '#f1c40f']
    
    params = []
    flops = []
    memory = []
    
    for name, model in model_dict.items():
        stats = summary(model, input_size=input_size, verbose=0)
        
        params.append(stats.total_params / 1e6)
        
        flops.append(stats.total_mult_adds / 1e9)
        
        total_bytes = stats.total_input + stats.total_output_bytes + stats.total_param_bytes
        memory.append(total_bytes / (1024**2))

    # Plotting Logic
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [params, flops, memory]
    titles = ['Total Parameters (Millions) ↓', 
              'Computational Cost (GFLOPs) ↓', 
              'Estimated Total Size (MB) ↓']
    y_labels = ['Millions', 'Giga-Operations', 'MB']

    for i in range(3):
        bars = axes[i].bar(model_names, metrics[i], color=colors[:len(model_names)])
        axes[i].set_title(titles[i], fontweight='bold')
        axes[i].set_ylabel(y_labels[i])
        
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()