import os
import time
import json
import fire
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser
import base64
import pandas as pd
import pyarrow.parquet as pq
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.common.datatable.data_table import DataTable

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from .Model import ESPNet_Encoder, ESPNet


class SSESPNet:
    def __init__(self, model_path='saved_model', meta={}):
        self.mean = [72.3923111, 82.90893555, 73.15840149]
        self.std = [45.3192215, 46.15289307, 44.91483307]
        self.inference_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.stdv),
        ])
        if meta['Model Type'] == 'ESPNet':
            self.model = ESPNet(meta['Classes'], p=meta['P'], q=meta['Q'])
        else:
            self.model = ESPNet_Encoder(meta['Classes'], p=meta['P'], q=meta['Q'])
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location='cpu'))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model).cuda()

        self.model.eval()
        self.classes = my_dict
        self.print_freq = 1

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

    def evaluate(self, data_path='test_data', save_path='outputs'):
        os.makedirs(save_path, exist_ok=True)
        input = pd.read_parquet(os.path.join(data_path, 'data.dataset.parquet'), engine='pyarrow')
        df = self.run(input)
        # df.to_parquet(fname=os.path.join(self.save_path, 'labels.parquet'), engine='pyarrow')
        # input = pd.read_parquet(os.path.join(self.save_path, 'labels.parquet'), engine='pyarrow')
        # print(input)
        dt = DataTable(df)
        OutputHandler.handle_output(data=dt, file_path=save_path,
                                    file_name='data.dataset.parquet', data_type=DataTypes.DATASET)


def test(args):
    meta = {'Model Type': args.model_type, 'P': args.p, 'Q': args.q, 'Classes': 20}
    ssespnet = SSESPNet(args.model_path, meta)
    ssespnet.evaluate(data_path=args.data_path, save_path=args.save_path)

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
    with open(os.path.join(args.save_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_type', default='ESPNet', help='Model name')
    parser.add_argument('--data_path', default='script/outputs', help='Data directory')
    parser.add_argument('--save_path', default='script/outputs2', help='directory to save the results')
    parser.add_argument('--model_path', default='script/saved_model', help='Model directory')
    parser.add_argument('-p', default=2, type=int, help='depth multiplier')
    parser.add_argument('-q', default=8, type=int, help='depth multiplier')

    test(parser.parse_args())
