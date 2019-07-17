import fire
import os
import torch
from torchvision import transforms
from PIL import Image

import torch.nn as nn
from .models import DenseNet


def inference(model_path, data_path='test_data', save_path='outputs'):

    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    inference_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

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


if __name__ == '__main__':
    fire.Fire(inference)
