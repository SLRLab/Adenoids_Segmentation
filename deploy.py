#from ppDeploy import *
# encoding = utf-8
from model import Segmentation_Model
import gradio as gr
from PIL import Image
import torch
import cv2 
import numpy as np
import torchvision.transforms as transforms


def seg_predict(img):
    img = cv2.resize(img, dsize = (448,448))
    img = img_transform(img)
    img = img.unsqueeze(0)
    output = seg(img)
    
    # tensor.detach().numpy()
    output = output.detach().numpy()[0].transpose(1,2,0)
    _, pred = cv2.threshold(output, 0.5, 1, 0)

    pre = np.zeros((448,448,3))

    pre[..., 1] = pred[:,:,1]
    pre[..., 0] = pred[:,:,0]

    pre = pre.astype('uint8')

    pre_show = np.zeros((448, 448, 3)) 
    pre_show[..., 2] = pre[:, :, 1]
    pre_show[..., 1] = pre[:, :, 2]

    return pre*255


if __name__ == '__main__':
    gr.close_all()
    
    seg = Segmentation_Model()
    model_path = 'xxx'
    state_dict = torch.load(model_path)
    seg.load_state_dict(state_dict['best_IoU1_param'])
    img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])])
    demo = gr.Interface(fn = seg_predict,
          inputs = gr.Image(label = "腺样体图像", shape = (448,448), height = 600, width = 600),
          outputs = gr.Image(label = "分割图像", shape = (448, 448), height = 600, width = 600),
          description = "腺样体诊断系统",
          allow_flagging = 'never',
          )
    demo.launch(server_name = "172.21.109.64")
