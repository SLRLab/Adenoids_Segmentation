import os
import torchvision.transformers as transformers


class LaryngoScope(Dataset):
    def __init__(self, dataset_path=None, is_aug=False, imgsize=448, kfold = 5, fold_num = 2, stage = 'train'):
        dataset_json = json.load(open(dataset_path, encoding = 'utf-8'))
        file_list = dataset_json['train_list'] + dataset_json['val_list']
        number = len(file_list)

        self.img_list = []
        self.mask_list = []
        self.train_size = imgsize
        self.is_aug = is_aug

        valSet = [file_list[i] for i in range((fold_num - 1) * int(number/kfold), fold_num * int(number/kfold))]
        trainSet = list(set(valSet) ^ set(file_list))
    
        img_path = 'xxx'
        mask_path = 'xxx'

        if stage == 'train':
            self.img_list = [os.path.join(img_path, i) for i in trainSet]
            self.mask_list = [os.path.join(mask_path, i) for i in trainSet]

        elif stage == 'val':
            self.img_list = [os.path.join(img_path, i) for i in valSet]
            self.mask_list = [os.path.join(mask_path, i) for i in valSet]

        # 增強操作
        if self.is_aug == True:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize((self.train_size, self.train_size)),
                #transforms.RandomRotation(90, resample=False, expand=False, center=None),
                #transforms.RandomVerticalFlip(p=0.5),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.Resize((self.train_size, self.train_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    #transforms.RandomRotation(90, resample=False, expand=False, center=None),
                    #transforms.RandomVerticalFlip(p=0.5),
                    #transforms.RandomHorizontalFlip(p=0.5),
                      #transforms.Resize((self.train_size, self.train_size)),
                    transforms.ToTensor()])

            else:
                self.img_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    #transforms.Resize((self.train_size, self.train_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    
                self.gt_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    #transforms.Resize((self.train_size, self.train_size)),
                    transforms.ToTensor()
                    ])

    def __len__(self):
        return len(self.img_list)
                                                                                                                                                                                               3,0-1         Top

