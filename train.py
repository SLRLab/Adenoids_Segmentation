import torch
import time
import torch.optim as optim
from dataloader import LaryngoScope
from model import Segmentation_Model
import torch.optim as optim
from loss import loss_function
from torch.utils.data import Dataset,DataLoader



if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_list = r'xxx'
    lr = 0.0001
    num_epoch = 100 
    k_fold = 5 
    model_path = r'...'

    for k in range(k_fold):
        res_path = r'xxx' + str(k) + '/' 
        weight_path = r'xxx' + str(k)  + '.pth'

        trainDataset = LaryngoScope(data_list, is_aug = True, imgsize = fix_size, kfold = 5, fold_num = k + 1, stage = 'train')
        valDataset = LaryngoScope(data_list, is_aug = False, imgsize = fix_size, kfold = 5, fold_num = k + 1, stage = 'val')

        train_loader = DataLoader(trainDataset, batch_size = batch_size, shuffle = True,
            num_workers = 8, pin_memory = True, drop_last = True)

        val_loader = DataLoader(valDataset, batch_size= batch_size, shuffle=False,
            num_workers=8, pin_memory=True, drop_last=True)

        model = Segmentation_Model().to('cuda:0')
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        more_metric = Numeric_score()

        best_IoU1 = 0 
        best_IoU2 = 0 
        best_Dice1 = 0 
        best_Dice2 = 0 

        best_IoU1_param = None
        best_IoU2_param = None
        best_Dice1_param = None
        best_Dice2_param = None
        for epoch in range(num_epoch):
            epoch_start_time = time.time()
            train_loss = 0.0
            test_loss = 0.0

            change = 0

            # train process
            print("========== train process ===========")
            model.train()

            for i, data in enumerate(train_loader):
                output = model(data[0].cuda(0))
                label = data[1]

                optimizer.zero_grad()
                batch_loss = loss_function(output, label.cuda(0))
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()

            # val process
            print("============= val process ==============")
            model.eval()
            with torch.no_grad():
                cls1_iou = []
                cls2_iou = []
                cls1_F1 = []
                cls2_F1 = []
                cls1_F2 = []
                cls2_F2 = []
                cls1_pre = []
                cls2_pre = []
                HD_cls1 = []
                HD_cls2 = []
                Dice_cls1 = []
                Dice_cls2 = []
                Acc_cls1 = []
                Acc_cls2 = []
                Spe_cls1 = []
                Spe_cls2 = []

                counter = 0
                pred = []
                for i, data in enumerate(val_loader):
                    output = model(data[0].cuda(0))
                    label = data[1]

                    batch_loss = loss_funtion(output, label.cuda(0))
                    test_loss += batch_loss.item()

                    IoU,F1_score,F2_score,precision,Dice,Accuracy,Spe = more_metric(output, label.cuda(0))

                    output = output.cpu().detach()
                    label = label.cpu().detach()

                    output[output>=0.5] = 1.0
                    output[output<0.5] = 0.0

                    cls1_iou.append(IoU[0])
                    cls2_iou.append(IoU[1])

                    cls1_F1.append(F1_score[0])
                    cls2_F1.append(F2_score[1])


                    Dice_cls1.append(Dice[0])
                    Dice_cls2.append(Dice[1])

                    Acc_cls1.append(Accuracy[0])
                    Acc_cls2.append(Accuracy[1])

                    Spe_cls1.append(Spe[0])
                    Spe_cls2.append(Spe[1])

                    #iwrite_img(res_path, output, data[0], label, str(i), Dice[0], Dice[1])

                IoU1 = np.mean(cls1_iou)
                IoU2 = np.mean(cls2_iou)
                F1_score1 = np.mean(cls1_F1)
                F1_score2 = np.mean(cls2_F1)
                F2_score1 = np.mean(cls1_F2)
                F2_score2 = np.mean(cls2_F2)
                precision1 = np.mean(cls1_pre)
                precision2 = np.mean(cls2_pre)

                Dice_1 = np.mean(Dice_cls1)
                Dice_2 = np.mean(Dice_cls2)

                Acc_1 = np.mean(Acc_cls1)
                Acc_2 = np.mean(Acc_cls2)
                Spe_1 = np.mean(Spe_cls1)
                Spe_2 = np.mean(Spe_cls2)

                if IoU1 > best_IoU1:
                    change = 1
                    best_IoU1 = IoU1
                    best_IoU1_param = model.state_dict()

                if IoU2 > best_IoU2:
                    change = 1
                    best_IoU2 = IoU2
                    best_IoU2_param = model.state_dict()

                if Dice_1 > best_Dice1:
                    change = 1
                    best_Dice1 = Dice_1
                    best_Dice1_param = model.state_dict()

                if Dice_2 > best_Dice2:
                    change = 1
                    best_Dice2 = Dice_2
                    best_Dice2_param = model.state_dict()

                if change == 1:
                    torch.save({
                    'best_IoU1': best_IoU1,
                    'best_IoU1_param': best_IoU1_param,
                    'best_IoU2': best_IoU2,
                    'best_IoU2_param': best_IoU2_param,
                    'best_Dice1': best_Dice1,
                    'best_Dice1_param': best_Dice1_param,
                    'best_Dice2': best_Dice2,
                    'best_Dice2_param': best_Dice2_param
                    }, weight_path)
                    change = 0

                print('[%03d/%03d] %2.2f sec(s)  Train Loss: %3.6f | Val loss: %3.6f' % (epoch + 1, num_epoch,
                time.time() - epoch_start_time, train_loss / (batch_size * len(train_set)), test_loss / (batch_size * len(val_set))))

                                                                                                                                                      
