import os
import json
import fire
import pandas as pd
import base64
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.common.datatable.data_table import DataTable


def entrance(data_path='script/test_data', save_path='script/outputs'):
    my_list = []
    image_list = os.listdir(data_path)
    post_list = ['jfif', 'png', 'jpg', 'jpeg']
    for file_name in image_list:
        lists = file_name.split('.')
        if lists[-1] not in post_list:
            continue
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'rb') as f:
            s = base64.b64encode(f.read())
        input_data = s.decode('ascii')
        input_data = 'data:image/png;base64,' + input_data
        my_list.append([input_data])
    df = pd.DataFrame(my_list, columns=['image_string'])
    os.makedirs(save_path, exist_ok=True)
    # df.to_parquet(fname=os.path.join(save_path, 'data.dataset.parquet'), engine='pyarrow')
    dt = DataTable(df)
    OutputHandler.handle_output(data=dt, file_path=save_path,
                                file_name='data.dataset.parquet', data_type=DataTypes.DATASET)

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
    fire.Fire(entrance)
