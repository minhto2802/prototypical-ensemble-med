import torch
from torchvision.transforms import v2


def get_augmentations(stage=0, dataset_name: str = None, input_size=224):
    match dataset_name:
        case 'celeba':
            random_resize_crop = v2.Compose([
                # v2.RandomCrop(178),
                # v2.CenterCrop(178),
                # v2.Resize((input_size, input_size))
                # v2.RandomHorizontalFlip(0.5),
                v2.Resize((256, 256)),
                v2.CenterCrop((input_size, input_size)),
            ])
            resize_crop = v2.Compose([
                # v2.CenterCrop(178),
                # v2.Resize((input_size, input_size))
                v2.Resize((256, 256)),
                v2.CenterCrop((input_size, input_size)),
            ])
        case _:
            random_resize_crop = v2.Compose([
                # v2.Resize((256, 256), antialias=True),
                v2.RandomResizedCrop(size=(input_size, input_size)),
                v2.RandomPhotometricDistort(p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
            ])
            resize_crop = v2.Compose([
                v2.Resize((256, 256), antialias=True),
                v2.CenterCrop((input_size, input_size)),
            ])

    transform = v2.Compose([
        random_resize_crop,
        # v2.RandomGrayscale(),
        # v2.RandomRotation(30),
        # v2.ToTensor(),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_eval = v2.Compose([
        resize_crop,
        # v2.ToTensor(),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if stage > 0:
        return {'train': transform_eval, 'eval': transform_eval}
    return {'train': transform, 'eval': transform_eval}
