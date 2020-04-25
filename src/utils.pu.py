def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()

