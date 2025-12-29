# GTR-Loc: Geospatial Text Regularization Assisted Outdoor LiDAR Localization

---

## Requirements
- python 3.9.17

- pytorch 2.0.1

- MinkowskiEngine 0.5.4

## Datasets

GTR-Loc currently supports the following datasets:
- Oxford Radar RobotCar
- QEOxford
- NCLT

We organize the dataset directories as follows:

- Oxford/QEOxford
```
data_root
├── 2019-01-11-14-02-26-radar-oxford-10k
│   ├── velodyne_left
│   │   ├── xxx.bin
│   │   ├── xxx.bin
│   │   ├── …
│   ├── velodyne_left_calibrateFalse.h5
│   ├── velodyne_left_False.h5
├── …
```
- NCLT
```
data_root
├── 2012-01-22
│   ├── velodyne_left
│   │   ├── xxx.bin
│   │   ├── xxx.bin
│   │   ├── …
    ├── velodyne_sync
│   │   ├── xxx.bin
│   │   ├── xxx.bin
│   │   ├── …
│   ├── velodyne_left_False.h5
├── …
```

## Train (Following LightLoc)

We follow LightLoc to train the network.

#### Oxford/QEOxford

- Train classifier 
```
follow LightLoc
```
- Train regressor
```
python train.py
```

#### NCLT

- Train classifier 
```
follow LightLoc
```
- Train regressor
```
python train.py --epochs=30 --prune_ratio=0.15 --level_cluster=100 --voxel_size=0.3
```
## Test
####  (QE)Oxford
```
python eval.py
```
####  NCLT
```
python eval.py --voxel_size=0.3
```
**Note:** For results matching Table 3 in the paper, exclude lines 4300–4500 from the 2012-05-26 trajectory.

## Acknowledgement

 We appreciate the code of LightLoc they shared.

## Citation

```
@inproceedings{YU2022108685,
title = {GTR-Loc: Geospatial Text Regularization Assisted Outdoor LiDAR Localization},
booktitle = {NeurIPS},
year = {2025},
author = {Shangshu Yu and Wen Li and Xiaotian Sun and Zhimin Yuan and Xin Wang and Sijie Wang and Rui She and Cheng Wang}
}
```
