# A Demo for Adenoid Segmentation
This repository includes the code for a model based Transformer for automatic segmentation of adenoid and nasal.

### Task Description
<img align="left" src="imgs/example1.png" title="Angular" hspace="10" width="150"/>
<img align="left" src="imgs/example1_mask.png"  title="Angular" hspace="20" width="150"/>
<img align="left" src="imgs/example2.jpg" title="Angular" hspace="10" width="150"/>
<img align="left" src="imgs/example2_mask.png" title="Angular" hspace="20" width="150"/>
<br/><br/><br/><br/><br/><br/><br/>
In the preview pictures, the left pictures are Adenoid endoscopic images. And the right pictures are masked images annotated by experienced doctors. In these pictures, red areas mean adenoid and green areas mean nasal.


### Environment
- OS Version: Ubuntu 16.04
- CUDA Version: 10.2
- torch Version: 1.71

### Network Overview
![Network](imgs/Network.png)




### File Description
- `dataloader.py`: a packaged dataloader for preprocessing adenoid images.
- `model.py`: The encoder comprises four Mix Transformer blocks with four down-sampling scales. The decoder combines feature maps from different scales. And the SE module is used to reweight the channel weights.
- `metric.py`: computing Acc, Spe, Dice, IoU, F1.
- `loss.py`: a simple loss function.
- `train.py`: the train process.
- `test.py`: the test process.
- `deploy.py`: using `gradio` to deploy the model into a website.
- `utils.py`: other functions.

