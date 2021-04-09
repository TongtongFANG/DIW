
# Dynamic Importance Weighting (DIW)

This is a reproducing code for Dynamic Importance Weighting (DIW) in the NeurIPS'20 paper: Rethinking Importance Weighting for Deep Learning under Distribution Shift.

Link to the paper: https://proceedings.neurips.cc//paper/2020/file/8b9e7ab295e87570551db122a04c6f7c-Paper.pdf

## Requirements
The code was developed and tested based on the following environment.
- python 3.8
- pytorch 1.6.0
- torchvision 0.7.0
- cudatoolkit 10.2
- cvxopt 1.2.0
- matplotlib 
- sklearn
- tqdm

## Quick start
You can run an example code of DIW on Fashion-MNIST under 0.4 symmetric label noise.

`python diw.py`

## Example result
After running `python diw.py`, a output figure and text file of test accurary are made in `./output/` by default. 

## Citation
If the code is useful in your research, please cite the following:  
Tongtong Fang, Nan Lu, Gang Niu, Masashi Sugiyama. Rethinking Importance Weighting for Deep Learning under Distribution Shift. NeurIPS 2020. 

