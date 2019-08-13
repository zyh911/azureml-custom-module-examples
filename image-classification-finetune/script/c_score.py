import os
import json
import fire
import pandas as pd
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.common.datatable.data_table import DataTable

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from .densenet import densenet201, densenet169, densenet161, densenet121, MyDenseNet
from .index_to_label import my_dict
from .imagenet1000_label_to_index import new_dict


class CScore:
    def __init__(self, compared_model_path, model_path, meta={}):
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

        self.model = MyDenseNet(model_type=meta['Model type'], pretrained=False,
                                memory_efficient=self.memory_efficient, classes=120)
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location='cpu'))

        if meta['Model type'] == 'densenet201':
            self.cmodel = densenet201(model_path=compared_model_path, pretrained=True,
                                      memory_efficient=self.memory_efficient)
        elif meta['Model type'] == 'densenet169':
            self.cmodel = densenet169(model_path=compared_model_path, pretrained=True,
                                      memory_efficient=self.memory_efficient)
        elif meta['Model type'] == 'densenet161':
            self.cmodel = densenet161(model_path=compared_model_path, pretrained=True,
                                      memory_efficient=self.memory_efficient)
        else:
            self.cmodel = densenet121(model_path=compared_model_path, pretrained=True,
                                      memory_efficient=self.memory_efficient)

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
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                temp_cnt = target.shape[0]
                total_cnt += temp_cnt

                output1 = self.model(input)
                output1 = torch.argmax(output1, 1)
                temp1 = torch.zeros([temp_cnt], dtype=torch.uint8)
                temp1[output1 == target] = 1
                true_cnt1 += torch.sum(temp1).item()

                for i in range(len(target)):
                    name = label_list[target[i]]
                    names = name.split('-', 1)
                    names2 = names[-1].split('_')
                    final_str = ' '.join(names2)
                    target[i] = torch.tensor(new_dict[final_str])

                output2 = self.cmodel(input)
                output2 = torch.argmax(output2, 1)
                temp2 = torch.zeros([temp_cnt], dtype=torch.uint8)
                temp2[output2 == target] = 1
                true_cnt2 += torch.sum(temp2).item()

        print(true_cnt1 / total_cnt, true_cnt2 / total_cnt)

        df = pd.DataFrame([[true_cnt1 / total_cnt, true_cnt2 / total_cnt]],
                          columns=['Model accuracy', 'Compared model accuracy'])
        return df

    def evaluate(self, data_path='test_data', save_path='outputs'):
        os.makedirs(save_path, exist_ok=True)
        df = self._evaluate_with_label(data_path)
        dt = DataTable(df)
        OutputHandler.handle_output(data=dt, file_path=save_path,
                                    file_name='data.dataset.parquet', data_type=DataTypes.DATASET)


def test(compared_model_path='script/saved_model', model_path='script/saved_model', data_path='script/dataset/dog_test',
         save_path='script/outputs2', model_type='densenet201', memory_efficient=False):
    meta = {'Model type': model_type, 'Memory efficient': str(memory_efficient)}
    cscore = CScore(compared_model_path, model_path, meta)
    cscore.evaluate(data_path=data_path, save_path=save_path)

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
