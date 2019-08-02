import os
import time
import json
import fire
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import pyarrow.parquet as pq
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.common.datatable.data_table import DataTable

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from .densenet import densenet201, densenet169, densenet161, densenet121
from .imagenet1000_index_to_label import my_dict


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
    def __init__(self, model_path='saved_model', meta={}):
        self.mean = [0.485, 0.456, 0.406]
        self.stdv = [0.229, 0.224, 0.225]
        self.inference_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.stdv),
        ])
        print(1)
        print(meta)
        if meta['Model Type'] == 'densenet201':
            self.model = densenet201(pretrained=False, memory_efficient=meta['Memory efficient'])
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'model201.pth'), map_location='cpu'))
        elif meta['Model Type'] == 'densenet169':
            self.model = densenet169(pretrained=False, memory_efficient=meta['Memory efficient'])
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'model169.pth'), map_location='cpu'))
        elif meta['Model Type'] == 'densenet161':
            self.model = densenet161(pretrained=False, memory_efficient=meta['Memory efficient'])
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'model161.pth'), map_location='cpu'))
        else:
            self.model = densenet121(pretrained=False, memory_efficient=meta['Memory efficient'])
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'model121.pth'), map_location='cpu'))

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model).cuda()

        self.model.eval()
        self.classes = my_dict
        self.print_freq = 1
        print(2)

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
        self.has_label = False
        try:
            self.test_set = datasets.CIFAR10(self.data_path, train=False,
                                             transform=self.inference_transforms, download=False)
            self.has_label = True
        except Exception as e:
            print(e)
            self.image_list = os.listdir(self.data_path)
        if self.has_label:
            self._evaluate_with_label()
        else:
            self._evaluate_without_label()

    def run(self, input, meta=None):
        my_list = []
        print(3)
        for i in range(input.shape[0]):
            print(4)
            temp_string = json.loads(input.iloc[i]['image_string'])
            if temp_string.startswith('data:'):
                my_index = temp_string.find('base64,')
                temp_string = temp_string[my_index + 7:]
            temp = base64.b64decode(temp_string)
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
            print(self.classes[index])
            my_list.append(result)
        # print(my_list)
        print(5)
        output = [[x['label'], x['probability']] for x in my_list]
        df = pd.DataFrame(output, columns=['label', 'probability'])
        return df

    def evaluate_new(self, data_path='test_data', save_path='outputs'):
        self.data_path = data_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        input = pd.read_parquet(os.path.join(self.data_path, 'data.dataset.parquet'), engine='pyarrow')
        df = self.run(input)
        # df.to_parquet(fname=os.path.join(self.save_path, 'labels.parquet'), engine='pyarrow')
        # input = pd.read_parquet(os.path.join(self.save_path, 'labels.parquet'), engine='pyarrow')
        # print(input)
        dt = DataTable(df)
        OutputHandler.handle_output(data=dt, file_path=self.save_path,
                                    file_name='data.dataset.parquet', data_type=DataTypes.DATASET)


def test(model_path='saved_model', data_path='outputs', save_path='outputs2', model_type='densenet201',
         memory_efficient=False):
    meta = {'model_type': model_type, 'memory_efficient': memory_efficient}
    icdensenet = ICDenseNet(model_path, meta)
    icdensenet.evaluate_new(data_path=data_path, save_path=save_path)

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        'Id': 'Dataset',
        'Name': 'Dataset .NET file',
        'ShortName': 'Dataset',
        'Description': 'A serialized DataTable supporting partial reads and writes',
        'IsDirectory': False,
        'Owner': 'Microsoft Corporation',
        'FileExtension': 'dataset.parquet',
        'ContentType': 'application/octet-stream',
        'AllowUpload': False,
        'AllowPromotion': True,
        'AllowModelPromotion': False,
        'AuxiliaryFileExtension': None,
        'AuxiliaryContentType': None
    }
    with open(os.path.join(save_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)


if __name__ == '__main__':
    fire.Fire(test)
