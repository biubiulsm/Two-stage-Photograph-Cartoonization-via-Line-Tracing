# Two-stage-Photograph-Cartoonization-via-Line-Tracing
## Paper
This is a pytorch implementation of Pacific Graphics 2020 paper "Two-stage Photograph Cartoonization via Line Tracing". 
![image](https://github.com/biubiulsm/Two-stage-Photograph-Cartoonization-via-Line-Tracing/blob/master/representative.jpg)

## Requirement

## Training
- During training, to view training loss plots and visual results, run `python -m visdom.server` and click the URL shown below the command. 
- To train the model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
Want see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.

## Testing
- To test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.


## Citation

