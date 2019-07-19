import os
import json
import fire
import pandas as pd
import base64


def entrance(data_path='dataset', save_path='outputs'):
    my_list = []
    image_list = os.listdir(data_path)
    for file_name in image_list:
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'rb') as f:
            s = base64.b64encode(f.read())
        input_data = json.dumps(s.decode('ascii'))
        my_list.append([input_data])
    df = pd.DataFrame(my_list, columns=['image_string'])
    df.to_parquet(fname=os.path.join(save_path, "image_data.parquet"))

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "ILearnerDotNet",
        "Name": "ILearner .NET file",
        "ShortName": "Model",
        "Description": "A .NET serialized ILearner",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "ilearner",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": False,
        "AllowModelPromotion": True,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(save_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)

    print('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
