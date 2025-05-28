import torch
import pickle

cp_path = 'results/h36m/models/0500_cpu.p'

model_cp = pickle.load(open(cp_path, "rb"))

def recursive_to_cuda(obj, device='cuda:0'):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: recursive_to_cuda(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_cuda(i, device) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_cuda(i, device) for i in obj)
    else:
        return obj

model_cp = recursive_to_cuda(model_cp, device='cuda:0')
