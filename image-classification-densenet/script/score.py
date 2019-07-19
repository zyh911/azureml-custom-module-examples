import os
import time
import json
import fire
from PIL import Image
from io import BytesIO
import base64

import torch
import torch.nn as nn
from torchvision import datasets, transforms

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


class ICDenseNet:
    def __init__(self, model_path='saved_model', data_path='test_data', save_path='outputs', print_freq=1):
        self.mean = [0.5071, 0.4867, 0.4408]
        self.stdv = [0.2675, 0.2565, 0.2761]
        self.inference_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.stdv),
        ])

        self.model = DenseNet()
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model).cuda()

        self.model.eval()
        self.data_path = data_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.has_label = False
        try:
            self.test_set = datasets.CIFAR10(self.data_path, train=False,
                                             transform=self.inference_transforms, download=False)
            self.has_label = True
            self.print_freq = print_freq
        except Exception as e:
            print(e)
            self.image_list = os.listdir(self.data_path)

    def _evaluate_with_label(self):
        test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=64, shuffle=False,
                                                  pin_memory=(torch.cuda.is_available()), num_workers=0)
        batch_time = AverageMeter()
        losses = AverageMeter()
        error = AverageMeter()

        end = time.time()
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(test_loader):
                # Create vaiables
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                # compute output
                output = self.model(input)
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
                if batch_idx % self.print_freq == 0:
                    res = '\t'.join([
                        'Iter: [{:d}/{:d}]'.format(batch_idx + 1, len(test_loader)),
                        'Avg_Time_Batch/Avg_Time_Epoch: {:.3f}/{:.3f}'.format(batch_time.val, batch_time.avg),
                        'Avg_Loss_Batch/Avg_Loss_Epoch: {:.4f}/{:.4f}'.format(losses.val, losses.avg),
                        'Avg_Error_Batch/Avg_Error_Epoch: {:.4f}/{:.4f}'.format(error.val, error.avg),
                    ])
                    print(res)

        print('batch_time.avg: {:.3f}, losses.avg: {:.4f}, error.avg: {:.4f}'.format(batch_time.avg, losses.avg, error.avg))
        with open(os.path.join(self.save_path, 'results.csv'), 'w') as f:
            f.write('batch_time.avg,losses.avg,error.avg\n')
            f.write('{:.5f},{:.5f},{:.5f}\n'.format(batch_time.avg, losses.avg, error.avg))

        # Return summary statistics
        print('This experiment has been completed.')
        return

    def _evaluate_without_label(self):
        with open(os.path.join(self.save_path, 'results.csv'), 'w') as f:
            f.write('image_name,predicted_class,probability\n')
            for file_name in self.image_list:
                file_path = os.path.join(self.data_path, file_name)
                img = Image.open(file_path)

                input_tensor = self.inference_transforms(img)
                input_tensor = input_tensor.unsqueeze(0)
                with torch.no_grad():
                    if torch.cuda.is_available():
                        input_tensor = input_tensor.cuda()
                    output = self.model(input_tensor)
                    softmax = nn.Softmax(dim=1)
                    pred_probs = softmax(output).cpu().numpy()[0]
                    index = torch.argmax(output, 1)[0].cpu().item()
                    print('image_name: {}, predicted_class: {}, probability: {:.5f}'
                          .format(file_name, self.classes[index], pred_probs[index]))
                    f.write('{},{},{:.5f}\n'.format(file_name, self.classes[index], pred_probs[index]))

        print('This experiment has been completed.')
        return

    def evaluate(self):
        if self.has_label:
            self._evaluate_with_label()
        else:
            self._evaluate_without_label()

    def run(self, input):
        my_list = []
        for input_data in input:
            temp = base64.b64decode(json.loads(input_data))
            img = Image.open(BytesIO(temp))
            input_tensor = self.inference_transforms(img)
            input_tensor = input_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            with torch.no_grad():
                output = self.model(input_tensor)
                softmax = nn.Softmax(dim=1)
                pred_probs = softmax(output).cpu().numpy()[0]
                index = torch.argmax(output, 1)[0].cpu().item()

            result = {'label': self.classes[index], 'probability': str(pred_probs[index])}
            my_list.append(result)
        return my_list


def test(model_path='saved_model', data_path='test_data', save_path='outputs', print_freq=1):
    icdensenet = ICDenseNet(model_path, data_path, save_path, print_freq)
    icdensenet.evaluate()

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "Dataset",
        "Name": "Dataset .NET file",
        "ShortName": "Dataset",
        "Description": "A serialized DataTable supporting partial reads and writes",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "dataset.parquet",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": True,
        "AllowModelPromotion": False,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(save_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)


if __name__ == '__main__':
    fire.Fire(test)
