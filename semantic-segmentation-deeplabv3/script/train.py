import os
import json
import fire
import torch

from .model import deeplabv3_resnet101, fcn_resnet101


def entrance_fake(model_type='deeplabv3_resnet101', model_path='script/saved_model', save_path='script/saved_model'):

    os.makedirs(save_path, exist_ok=True)

    if model_type == 'deeplabv3_resnet101':
        model = deeplabv3_resnet101(model_path=model_path, pretrained=True)
    else:
        model = fcn_resnet101(model_path=model_path, pretrained=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        'Id': 'ILearnerDotNet',
        'Name': 'ILearner .NET file',
        'ShortName': 'Model',
        'Description': 'A .NET serialized ILearner',
        'IsDirectory': False,
        'Owner': 'Microsoft Corporation',
        'FileExtension': 'ilearner',
        'ContentType': 'application/octet-stream',
        'AllowUpload': False,
        'AllowPromotion': False,
        'AllowModelPromotion': True,
        'AuxiliaryFileExtension': None,
        'AuxiliaryContentType': None
    }
    with open(os.path.join(save_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)
    # Dump data.ilearner as a work around until data type design
    visualization = os.path.join(save_path, 'data.ilearner')
    with open(visualization, 'w') as file:
        file.writelines('{}')
    print('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance_fake)
