import os
import pickle
import time
from argparse import ArgumentParser
import json

import torch
import torch.optim.lr_scheduler
import torch.backends.cudnn as cudnn

from .loadData import LoadData
from .Model import ESPNet, ESPNet_Encoder
from .Criteria import CrossEntropyLoss2d
import script.Transforms as myTransforms
from .DataSet import MyDataset
from .IOUEval import iouEval


def val(args, val_loader, model, criterion):
    """
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    """
    # switch to evaluation mode
    model.eval()

    iouEvalVal = iouEval(args.classes)

    epoch_loss = []

    total_batches = len(val_loader)
    for i, (input_var, target_var) in enumerate(val_loader):
        start_time = time.time()

        if args.use_gpu and torch.cuda.is_available():
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        target_var[target_var == 255] = 19

        # run the mdoel
        output = model(input_var)

        # compute the loss
        loss = criterion(output, target_var)

        epoch_loss.append(loss.data.item())

        time_taken = time.time() - start_time

        # compute the confusion matrix
        iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.data.item(), time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    :param args: general arguments
    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    """
    # switch to train mode
    model.train()

    iouEvalTrain = iouEval(args.classes)

    epoch_loss = []

    total_batches = len(train_loader)
    for i, (input_var, target_var) in enumerate(train_loader):
        start_time = time.time()

        if args.use_gpu and torch.cuda.is_available():
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        target_var[target_var == 255] = 19

        # run the mdoel
        output = model(input_var)

        # set the grad to zero
        optimizer.zero_grad()
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data)
        time_taken = time.time() - start_time

        # compute the confusion matrix
        iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.data, time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU


def netParams(model):
    """
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


def trainValidateSegmentation(args):
    """
    Main function for training and validation
    :param args: global arguments
    :return: None
    """
    # check if processed data file exists or not
    if not os.path.isfile(args.cached_data_file):
        dataLoad = LoadData(args.data_path, args.classes, args.cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))

    q = args.q
    p = args.p
    # load the model
    if args.model_type == 'ESPNet-C':
        scaleIn = 8
        model = ESPNet_Encoder(args.classes, p=p, q=q)
    else:
        scaleIn = 1
        if args.pretrained:
            model = ESPNet(args.classes, p=p, q=q,
                           encoderFile=os.path.join(args.model_path, 'espnet_p_' + str(p) + '_q_' + str(q) + '.pth'))
        else:
            model = ESPNet(args.classes, p=p, q=q)

    if args.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()

    # create the directory if not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch
    if args.use_gpu and torch.cuda.is_available():
        weight = weight.cuda()

    criteria = CrossEntropyLoss2d(weight)  # weight

    if args.use_gpu and torch.cuda.is_available():
        criteria = criteria.cuda()

    print('Data statistics')
    print(data['mean'], data['std'])
    print(data['classWeights'])

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(1024, 512),
        myTransforms.RandomCropResize(32),
        myTransforms.RandomFlip(),
        # myTransforms.RandomCrop(64).
        myTransforms.ToTensor(scaleIn),
    ])

    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(1536, 768),  # 1536, 768
        myTransforms.RandomCropResize(100),
        myTransforms.RandomFlip(),
        # myTransforms.RandomCrop(64),
        myTransforms.ToTensor(scaleIn),
    ])

    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(1280, 720),  # 1536, 768
        myTransforms.RandomCropResize(100),
        myTransforms.RandomFlip(),
        # myTransforms.RandomCrop(64),
        myTransforms.ToTensor(scaleIn),
    ])

    trainDataset_scale3 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(768, 384),
        myTransforms.RandomCropResize(32),
        myTransforms.RandomFlip(),
        # myTransforms.RandomCrop(64),
        myTransforms.ToTensor(scaleIn),
    ])

    trainDataset_scale4 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(512, 256),
        # myTransforms.RandomCropResize(20),
        myTransforms.RandomFlip(),
        # myTransforms.RandomCrop(64),
        myTransforms.ToTensor(scaleIn),
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(1024, 512),
        myTransforms.ToTensor(scaleIn),
    ])

    # since we training from scratch, we create data loaders at different scales
    # so that we can generate more augmented data and prevent the network from overfitting

    trainLoader = torch.utils.data.DataLoader(
        MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main),
        batch_size=args.batch_size + 2, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale1),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainLoader_scale2 = torch.utils.data.DataLoader(
        MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale2),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainLoader_scale3 = torch.utils.data.DataLoader(
        MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale3),
        batch_size=args.batch_size + 4, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainLoader_scale4 = torch.utils.data.DataLoader(
        MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale4),
        batch_size=args.batch_size + 4, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        MyDataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=args.batch_size + 4, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.use_gpu and torch.cuda.is_available():
        cudnn.benchmark = True

    start_epoch = 0

    logFileLoc = os.path.join(args.save_path, args.logFile)
    logger = open(logFileLoc, 'w')
    logger.write("Parameters: %s" % (str(total_paramters)))
    logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # we step the loss by 2 after step size is reached
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.5)

    for epoch in range(start_epoch, args.max_epochs):

        scheduler.step(epoch)
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))

        # train for one epoch
        # We consider 1 epoch with all the training data (at different scales)
        train(args, trainLoader_scale1, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale2, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale4, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale3, model, criteria, optimizer, epoch)
        lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = train(args, trainLoader, model, criteria,
                                                                                   optimizer, epoch)

        # evaluate on validation set
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(args, valLoader, model, criteria)

        # save the model
        if args.use_gpu and torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), os.path.join(args.save_path, 'model.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pth'))

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val))
    logger.close()
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
    with open(os.path.join(args.save_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)
    # Dump data.ilearner as a work around until data type design
    visualization = os.path.join(args.save_path, 'data.ilearner')
    with open(visualization, 'w') as file:
        file.writelines('{}')
    print('This experiment has been completed.')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_type', default='ESPNet', help='Model name')
    parser.add_argument('--data_path', default='script/dataset/city_small', help='Data directory')
    parser.add_argument('--model_path', default='script/pretrained/encoder', help='Model directory')
    parser.add_argument('--max_epochs', type=int, default=300, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                  'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--save_path', default='script/saved_model', help='directory to save the results')
    parser.add_argument('--classes', type=int, default=20, help='No of classes in the dataset. 20 for cityscapes')
    parser.add_argument('--cached_data_file', default='city.p', help='Cached file name')
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--pretrained', default=False, type=bool,
                        help='Whether using pretrained models.'
                             'Only used when training ESPNet.')
    parser.add_argument('--use_gpu', default=False, type=bool)
    parser.add_argument('-p', default=2, type=int, help='depth multiplier')
    parser.add_argument('-q', default=8, type=int, help='depth multiplier')

    trainValidateSegmentation(parser.parse_args())
