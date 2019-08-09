import os
import time
import numpy as np
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
from .imagenet1000_label_to_index import new_dict


class ICDenseNet:
    def __init__(self, compared_model_path='saved_model', model_path='saved_model', meta={}):
        self.mean = [0.485, 0.456, 0.406]
        self.stdv = [0.229, 0.224, 0.225]
        self.inference_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.stdv),
        ])
        if meta['Memory efficient'] == 'True':
            self.memory_efficient = True
        else:
            self.memory_efficient = False
        if meta['Model Type'] == 'densenet201':
            self.model = densenet201(pretrained=False, memory_efficient=self.memory_efficient)
            self.cmodel = densenet201(pretrained=True, memory_efficient=self.memory_efficient)
        elif meta['Model Type'] == 'densenet169':
            self.model = densenet169(pretrained=False, memory_efficient=self.memory_efficient)
            self.cmodel = densenet169(pretrained=True, memory_efficient=self.memory_efficient)
        elif meta['Model Type'] == 'densenet161':
            self.model = densenet161(pretrained=False, memory_efficient=self.memory_efficient)
            self.cmodel = densenet161(pretrained=True, memory_efficient=self.memory_efficient)
        else:
            self.model = densenet121(pretrained=False, memory_efficient=self.memory_efficient)
            self.cmodel = densenet121(pretrained=True, memory_efficient=self.memory_efficient)
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location='cpu'))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.cmodel = self.cmodel.cuda()
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model).cuda()
                self.cmodel = torch.nn.DataParallel(self.cmodel).cuda()

        self.model.eval()
        self.cmodel.eval()
        self.classes = my_dict

    def _evaluate_with_label(self, data_path):
        test_set = datasets.ImageFolder(data_path, transform=self.inference_transforms)
        label_list = test_set.classes
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False,
                                                  pin_memory=(torch.cuda.is_available()), num_workers=0)
        total_cnt = 0
        true_cnt1 = 0
        true_cnt2 = 0

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(test_loader):
                # Create vaiables
                for i in range(len(target)):
                    name = label_list[target[i]]
                    names = name.split('-', 1)
                    names2 = names[-1].split('_')
                    final_str = ' '.join(names2)
                    target[i] = torch.tensor(new_dict[final_str])
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                output1 = self.model(input)
                output1 = torch.argmax(output1, 1)
                output2 = self.cmodel(input)
                output2 = torch.argmax(output2, 1)

                temp_cnt = output1.shape[0]
                total_cnt += temp_cnt
                temp1 = torch.zeros([temp_cnt], dtype=torch.uint8)
                temp2 = torch.zeros([temp_cnt], dtype=torch.uint8)
                temp1[output1 == target] = 1
                temp2[output2 == target] = 1
                true_cnt1 += torch.sum(temp1).item()
                true_cnt2 += torch.sum(temp2).item()
                for i in range(temp_cnt):
                    if output1[i] == target[i]:
                        true_cnt1 += 1
                    if output2[i] == target[i]:
                        true_cnt2 += 1

        print(true_cnt1 / total_cnt, true_cnt2 / total_cnt)

        df = pd.DataFrame([[true_cnt1 / total_cnt, true_cnt2 / total_cnt]],
                          columns=['Model accuracy', 'Compared model accuracy'])
        return df

    def run(self, input, meta=None):
        my_list = []
        for i in range(input.shape[0]):
            temp_string = input.iloc[i]['image_string']
            if temp_string.startswith('data:'):
                my_index = temp_string.find('base64,')
                temp_string = temp_string[my_index+7:]
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
        output = [[x['label'], x['probability']] for x in my_list]
        df = pd.DataFrame(output, columns=['label', 'probability'])
        return df

    def evaluate_new(self, data_path='test_data', save_path='outputs'):
        os.makedirs(save_path, exist_ok=True)
        df = self._evaluate_with_label(data_path)
        dt = DataTable(df)
        OutputHandler.handle_output(data=dt, file_path=save_path,
                                    file_name='data.dataset.parquet', data_type=DataTypes.DATASET)


def test(compared_model_path='script/saved_model', model_path='script/saved_model', data_path='script/dataset/dog_test',
         save_path='script/outputs2', model_type='densenet201', memory_efficient=False):
    meta = {'Model Type': model_type, 'Memory efficient': str(memory_efficient)}
    icdensenet = ICDenseNet(compared_model_path, model_path, meta)
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

    print('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(test)
