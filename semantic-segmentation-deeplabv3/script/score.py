import os
import json
import numpy as np
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
from torchvision import transforms

from .model import deeplabv3_resnet101


class SSDeeplabv3:
    def __init__(self, model_path='saved_model', meta={}):
        self.pallete = [
            [255, 255, 255],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0]
        ]

        self.mean = [0.485, 0.456, 0.406]
        self.stdv = [0.229, 0.224, 0.225]
        self.inference_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.stdv),
        ])

        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location='cpu'))
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()

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
                output = self.model(input_tensor)['out'][0]
                output_predictions = output.argmax(0).cpu().numpy()
                resultImg = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
                for idx in range(len(self.pallete)):
                    resultImg[output_predictions == idx] = self.pallete[idx]

                resultImg1 = Image.fromarray(resultImg).resize(img.size)
                resultImg2 = Image.blend(img, resultImg1, 0.5)
                imgByteArr1 = BytesIO()
                imgByteArr2 = BytesIO()
                resultImg1.save(imgByteArr1, format='PNG')
                imgBytes1 = imgByteArr1.getvalue()
                s1 = base64.b64encode(imgBytes1)
                s1 = s1.decode('ascii')
                resultImg2.save(imgByteArr2, format='PNG')
                imgBytes2 = imgByteArr2.getvalue()
                s2 = base64.b64encode(imgBytes2)
                s2 = s2.decode('ascii')
                s1 = 'data:image/png;base64,' + s1
                s2 = 'data:image/png;base64,' + s2

            my_list.append([s1, s2])
        df = pd.DataFrame(my_list, columns=['mask', 'fusion'])
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
    meta = {}
    ssespnet = SSDeeplabv3(args.model_path, meta)
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
    parser.add_argument('--data_path', default='script/outputs', help='Data directory')
    parser.add_argument('--save_path', default='script/outputs2', help='directory to save the results')
    parser.add_argument('--model_path', default='script/saved_model', help='Model directory')

    test(parser.parse_args())
