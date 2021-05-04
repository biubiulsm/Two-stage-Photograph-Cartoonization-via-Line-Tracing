# Two-stage-Photograph-Cartoonization-via-Line-Tracing
## Paper
This is a pytorch implementation of Pacific Graphics 2020 paper "Two-stage Photograph Cartoonization via Line Tracing". 
![image](https://github.com/biubiulsm/Two-stage-Photograph-Cartoonization-via-Line-Tracing/blob/master/representative.jpg)

## Requirement
```
Ubuntu 16.04 LTS
CPU or NVIDIA GPU + CUDA CuDNN
python=3.7.9
pytorch=1.7.1
torchvision=0.8.2
visdom=0.1.85
dominate=2.4.0
pillow=8.0.1
numpy=1.19.2
scipy=1.1.0
```

## Training
- During training, to view training loss plots and visual results, run `python -m visdom.server` and click the URL shown below the command. 
- To train the model:
```bash
python3 train.py --dataroot ./datasets/training_datasets --name Two_Stage_Cartoonization --model Two_Stage_Cartoonization --dataset_mode unaligned101 --gpu_ids 0 --resize_or_crop none
```
Want to see more intermediate results, check out `./checkpoints/Two_Stage_Cartoonization/web/index.html`.

## Testing
- To test the model:
```bash
python3 test.py --dataroot ./datasets/testing_datasets --name Two_Stage_Cartoonization --model Two_Stage_Cartoonization --dataset_mode single --gpu_ids 0 --resize_or_crop none --results_dir ./results/
```
- The testing results will be saved to the folder: `./results/Two_Stage_Cartoonization/latest_test`.
- The pretrained models are available at Google Drive: https://drive.google.com/drive/folders/18jrK3Aw0Yx9ZueTQw_yZrhJLedBAyspO?usp=sharing

## Citation
```
@inproceedings{li2020two,
  title={Two-stage Photograph Cartoonization via Line Tracing},
  author={Li, Simin and Wen, Qiang and Zhao, Shuang and Sun, Zixun and He, Shengfeng},
  booktitle={Computer Graphics Forum},
  year={2020},
  organization={Wiley Online Library}
}
```
