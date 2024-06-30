import torch
from dataloader import LaryngoScope
from model import Segmentation_Model
from torch.utils.data import Dataset
from metric import Number_score
from loss import loss_function



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    k = 0 
    res_path = r'xxx' + str(k) + '/' 
    weight_path = r'xxx' + str(k) + '.pth'

    valDataset = LaryngoScope(data_list, is_aug = False, imgsize = fix_size, kfold = 5, fold_num = k + 1, stage = 'val')

    val_loader = DataLoader(valDataset, batch_size= batch_size, shuffle=False,
                    num_workers=8, pin_memory=True, drop_last=True)

    model = Segmentation_Model().to('cuda:0')

    checkpoint = torch.load(weight_path)
    more_metric = Numeric_score()
    model.load_state_dict(checkpoint['best_IoU1_param'])
    
    test_loss = 0.0 

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
        
        
        for i, data in enumerate(val_loader):
            output = model(data[0].cuda(0))
            label = data[1]
        
            batch_loss = loss_function(output, label.cuda(0))
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
        cls1_pre.append(precision[0])
        cls2_pre.append(precision[1])

        Dice_cls1.append(Dice[0])
        Dice_cls2.append(Dice[1])

        Acc_cls1.append(Accuracy[0])
        Acc_cls2.append(Accuracy[1])

        Spe_cls1.append(Spe[0])
        Spe_cls2.append(Spe[1])

        #if  0.70 < IoU[1] < 0.75:
        #    print(str(i) + ': ' + str(IoU[1]))
        write_img(res_path, output, data[0], label, str(i), Dice[0], Dice[1])

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

        print('IoU_cls1: %.4f,IoU_cls2: %.4f'%(IoU1,IoU2))
        print('F1_cls1: %.4f,F1_cls2: %.4f'%(F1_score1,F1_score2))
        print('F2_cls1: %.4f,F2_cls2: %.4f'%(F1_score1,F1_score2))
        print('Pre_cls1: %.4f,Pre_cls2: %.4f'%(precision1,precision2))
        print('Dice_cls1: %.4f,Dice_cls2: %.4f'%(Dice_1,Dice_2))
        print('Acc_cls1: %.4f,Acc_cls2: %.4f'%(Acc_1,Acc_2))
        print('Spe_cls1: %.4f,Spe_cls2: %.4f'%(Spe_1,Spe_2))

