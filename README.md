# Two-stage-Photograph-Cartoonization-via-Line-Tracing
## Paper
This is a pytorch implementation of Pacific Graphics 2020 paper "Two-stage Photograph Cartoonization via Line Tracing". 
![image](https://github.com/biubiulsm/Two-stage-Photograph-Cartoonization-via-Line-Tracing/blob/master/representative.jpg)

## Requirement

## Training
- During training, to view training loss plots and visual results, run `python -m visdom.server` and click the URL shown below the command. 
- To train the model:
```bash
python3 train_fla3.py --dataroot ./datasets/training_datasets --name Two_Stage_Cartoonization --model Two_Stage_Cartoonization --dataset_mode unaligned101 --gpu_ids 0 --resize_or_crop none
```
Want to see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.

## Testing
- To test the model:
```bash
python3 test.py --dataroot ./datasets/testing_datasets --name Two_Stage_Cartoonization --model Two_Stage_Cartoonization --dataset single --resize_or_crop none --results_dir ./results/ --dataset_mode single 
```
- The testing results will be saved to the folder: `./results/maps_cyclegan/latest_test`.

## Citation

