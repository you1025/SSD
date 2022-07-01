import torch

def od_collate_fn(batch):
    images = []
    targets = []
    for (image, target) in batch:
        images.append(image)
        targets.append(torch.tensor(target, dtype=torch.float32))

    return (torch.stack(images, dim=0), targets)
