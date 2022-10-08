# PAPNet
Code for "Classification of Single-View Object Point Clouds" 

## Requirements
This code has been tested with
- python 3.7.12
- pytorch 1.6.0
- CUDA 10.1
- se3cnn [[link](https://github.com/mariogeiger/se3cnn/tree/546bc682887e1cb5e16b484c158c05f03377e4e9)]

## Downloads
- PartialModelNet40(PM40) dataset [[link](https://drive.google.com/file/d/1VWiAfkDAQeNortT9t_BcyJuvTb12gbhG/view?usp=sharing)]
- PartialScanNet15(PS15) dataset [[link](https://drive.google.com/file/d/1FO1RNNzYcAQwDngvDySEKYd-TQ-151OZ/view?usp=sharing)]
- Pretrained models of PAPNet [[link](https://drive.google.com/drive/folders/1HAQvS3PD9pEypJR3I3vssV-G_W_tiYBl?usp=sharing)]
- PartialModelNet40(PM40) under different levels of partiality [[link](https://drive.google.com/file/d/1ebXVYZkRjMvZ0oEgYMK1CHSSN5powu1_/view?usp=sharing)]

## Training
Command for training PAPNet:
```
python train.py --dataset pm40 --data_path path_to_pm40_dataset
python train.py --dataset ps15 --data_path path_to_ps15_dataset
```

## Evaluation
Evaluate the results of PAPNet:
```
python test.py --dataset [pm40|ps15] --data_path path_to_pm40_or_ps15 --model path_to_pretrained_model
python transfer_test_pm2ps.py --pm40_model path_to_pm40_model --ps15_path path_to_ps15_dataset
python transfer_test_ps2pm.py --ps15_model path_to_ps15_model --pm40_path path_to_pm40_dataset
```

## License
Our code is released under MIT License (see LICENSE file for details).
