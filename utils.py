import numpy as np



def gray2rgb(image):
    h, w, c = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    return image * 255



def write_img(res_path, predict, image, gt, filename, dice1, dice2):
    # (448,448,2) -> ()
    img = image.numpy()[0].transpose(1,2,0)
    gt = gt.numpy()[0].transpose(1,2,0)
    gt_show = np.zeros((448, 448, 3))
    gt_show[..., 2] = gt[:,:,0]
    gt_show[..., 1] = gt[:,:,1]
    pred_trans=predict.numpy()[0].transpose(1,2,0)
    _, pred_trans_ = cv2.threshold(pred_trans, 0.5, 1, 0)
    pred_result_ = np.zeros((448,448, 3))

    pred_result_[..., 2] = pred_trans_[:,:,1]
    pred_result_[..., 1] = pred_trans_[:,:,0]

    img = gray2rgb(img).copy()
    img = img.astype('uint8')

    cv2.imwrite(os.path.join(res_path, filename + 'img.jpg'), img)
    gt_show = gt_show.astype('uint8')
    pred_result_ = pred_result_.astype('uint8')


    cv2.imwrite(os.path.join(res_path, filename + 'pre.png'), img)
    pred_show = np.zeros((448, 448, 3))
    pred_show[..., 2] = pred_result_[:,:,1]
    pred_show[..., 1] = pred_result_[:,:,2]
    cv2.imwrite(os.path.join(res_path, filename + 'pre_mask.png'), pred_show*255)
    cv2.imwrite(os.path.join(res_path, filename + 'true_mask.png'), pred_show*255)
