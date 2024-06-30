import torch

class Numeric_score(nn.Module):
    def __init__(self):
        super(Numeric_score, self).__init__()

    def forward(self, prediction, groundtruth):
        C = groundtruth.size(1)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction >= 0.5).to(torch.float32)
        smooth = 1e-5
        beta = 2 

        IoU = []
        precision = []
        F1_score = []
        F2_score = []
        HD = []
        Dice = []
        Accuracy = []
        Spe = []

        for i in range(C):
    
            TP = int(((prediction[:,i] != 0) * (groundtruth[:,i] != 0)).sum())
            FP = int(((prediction[:,i] != 0) * (groundtruth[:,i] == 0)).sum())
            TN = int(((prediction[:,i] == 0) * (groundtruth[:,i] == 0)).sum())
            FN = int(((prediction[:,i] == 0) * (groundtruth[:,i] != 0)).sum())

            N = FP + FN + TP + TN
            T = int((groundtruth[:,i] == 1).sum())
            P = int((prediction[:,i] == 1).sum())
            Rec = TP/(T+smooth)
            Prec = TP/(P+smooth)

            iou = TP /(TP + FP + FN + smooth)
            dice = 2*TP/(2*TP + FP + FN + smooth)
            accuracy = (TP + TN )/ (N + smooth)
            pre = TP/(TP + FP + smooth)
            recall = TP/(TN + FP + smooth)
            specificity = TN / (TN + FP + smooth)
            f1score = 2*pre*recall/(pre + recall + smooth)
            f2score = 5 * pre * recall / (4 * pre + recall + smooth)
            hd_cls = Rec*Prec*(1+beta**2)/(Rec+beta**2*Prec+smooth)

            IoU.append(iou)
            F1_score.append(f1score)
            F2_score.append(f2score)
            precision.append(pre)
            HD.append(hd_cls)
            Dice.append(dice)
            Accuracy.append(accuracy)
            Spe.append(specificity)

        return IoU,F1_score,F2_score,precision,Dice,Accuracy,Spe
                                                                                                                                                                                                 1,1           Top

