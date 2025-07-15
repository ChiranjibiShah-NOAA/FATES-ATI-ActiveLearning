  

Active Learning Based Fish Species Identification
----------------------

### Requirements
For setup and data preparation, please refer to the README in [SSD pytorch](https://github.com/amdegroot/ssd.pytorch).

Code was tested in virtual environment with `Python 3+` and `Pytorch 1.1`.


Sample installation Steps: 
```
conda create --name alenv python=3.8
```
```
conda activate alenv
```
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```
```
pip install pycocotools
```
```
pip install opencv-python
```
This installation was tested in Nvidia A100 and Nvidia RTX(TM) 4090. 

### Setting Up Datasets

The dataset should be in the Pascal VOC dataset format. The fish dataset comprises 145 classes. You can change the number of classes from the config file.
 The PASCAL VOC folder structure is like this:
```
VOC_ROOT
|__ VOC2007
    |_ JPEGImages
      |_ xxx.jpg
    |_ Annotations
      |_ xxx.xml
    |_ ImageSets
       |_ Main
          |_ train.txt
          |_ val.txt
          |_ trainval.txt
          |_ test.txt
```


Training
--------
- Make directory `mkdir weights` and `cd weights`.

- Download the [FC-reduced VGG-16 backbone weight](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth) in the `weights` directory, and `cd ..`.

- If necessary, change the `VOC_ROOT` in `data/voc0712.py` .

- Please refer to `data/config.py` for configuration.

- Run the training code:

```
# Active learning
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_ssd_gmm_active_learining.py
```


Evaluation
--------
- Run the evaluation code:
```
# Evaluation on PASCAL VOC
python eval_voc.py --trained_model <trained weight path>
```


Visualization
---------
- Run the visualization code:
```
python demo.py --trained_model <trained weight path>
```

### References
This work is adapted from [AL-MDN](https://github.com/NVlabs/AL-MDN), and [SSD](https://github.com/lufficc/SSD/tree/master).


If you find this work useful, please feel free to cite:
```bash
@inproceedings{nabi2023probabilistic,
  title={Probabilistic Model-Based Active Learning with Attention Mechanism for Fish Species Recognition},
  author={Nabi, MM and Shah, Chiranjibi and Alaba, Simegnew Yihunie and Prior, Jack and Campbell, Matthew D and Wallace, Farron and Moorhead, Robert and Ball, John E},
  booktitle={OCEANS 2023-MTS/IEEE US Gulf Coast},
  pages={1--8},
  year={2023},
  organization={IEEE}
}
```


This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
