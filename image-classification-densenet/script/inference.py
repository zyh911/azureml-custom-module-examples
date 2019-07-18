import fire
import os
import torch
import time
from torchvision import datasets, transforms
from PIL import Image

import torch.nn as nn
from .densenet import DenseNet


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def inference(model_path, data_path='test_data', save_path='outputs'):

    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    inference_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    print(os.getcwd())
    model = DenseNet()
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()

    model.eval()
    os.makedirs(save_path, exist_ok=True)
    image_list = os.listdir(data_path)
    for file_name in image_list:
        name_raw = file_name.split('.')[0]
        file_path = os.path.join(data_path, file_name)
        img = Image.open(file_path)

        input_tensor = inference_transforms(img)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            output = model(input_tensor)
            output_label = torch.argmax(output.cpu()[0]).item()
            print(output_label)
            with open(os.path.join(save_path, name_raw + '.txt'), 'w') as f:
                f.write(str(output_label))

    print('This experiment has been completed.')


def test(model_path='models', data_path='dataset', save_path='outputs', print_freq=1):

    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    inference_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    print(os.getcwd())
    model = DenseNet()
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()

    test_set = datasets.CIFAR10(data_path, train=False, transform=inference_transforms, download=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Iter: [{:d}/{:d}]'.format(batch_idx + 1, len(test_loader)),
                    'Avg_Time_Batch/Avg_Time_Epoch: {:.3f}/{:.3f}'.format(batch_time.val, batch_time.avg),
                    'Avg_Loss_Batch/Avg_Loss_Epoch: {:.4f}/{:.4f}'.format(losses.val, losses.avg),
                    'Avg_Error_Batch/Avg_Error_Epoch: {:.4f}/{:.4f}'.format(error.val, error.avg),
                ])
                print(res)

    # Return summary statistics
    print('This experiment has been completed.')
    return batch_time.avg, losses.avg, error.avg


if __name__ == '__main__':
    fire.Fire(test)
