### lyft-3d-det-second-pointpillars

### references / citations

SECOND: Sparsely Embedded Convolutional Detection<br>
by Yan Yan, Yuxing Mao, and Bo Li<br>
https://www.semanticscholar.org/paper/SECOND%3A-Sparsely-Embedded-Convolutional-Detection-Yan-Mao/5125a16039cabc6320c908a4764f32596e018ad3<br>
https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf<br>
https://github.com/traveller59/second.pytorch<br>

PointPillars: Fast Encoders for Object Detection from Point Clouds<br>
by Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom<br>
https://arxiv.org/abs/1812.05784<br>
https://arxiv.org/pdf/1812.05784.pdf<br>
https://github.com/nutonomy/second.pytorch/blob/master/second/pytorch/models/pointpillars.py<br>

### steps to run

Note: These instructions have been verified on Ubuntu 20.04, if you are using something different you will have to make adjustments as applicable

#### 1) install and configure Ubuntu 20.04

Install Ubuntu 20.04 and configure to work with an NVIDIA GPU, also install PyTorch LTS 1.8.2.  Following this document is recommended: https://github.com/cdahms123/UbuntuSetup

#### 2) installations

```
sudo apt-get install libboost-all-dev

sudo apt-get install cmake

sudo -H pip3 install numba
sudo -H pip3 install scikit-image
sudo -H pip3 install seaborn
sudo -H pip3 install tensorboardX
sudo -H pip3 install psutil
sudo -H pip3 install pccm
sudo -H pip3 install plotly
sudo -H pip3 install lyft-dataset-sdk
```
install traveller59's `spconv` (checked out to specific commit for v1.2.1 Jan 30th, 2021)
```
cd ~
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv
git checkout fad3000249d27ca918f2655ff73c41f39b0f3127
export PATH=$PATH:$CUDA_HOME/bin
git submodule update --init --recursive
python3 setup.py bdist_wheel
cd dist
sudo -H pip3 install spconv-1.2.1-cp38-cp38-linux_x86_64.whl
pip3 list | grep spconv     # verify spconv is installed
```

create a symlink from `python3` to `python`
```
cd /usr/bin
sudo ln -s python3 python

# verify symlink creation
ls -l | grep python
# there should now be a link:
# python -> python3
```

#### 3) dataset download and setup

download the Kaggle Lyft Level 5 Object Detection dataset from https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data and arrange as follows:
```
~ (home directory)
    |-- LyftObjDetDataset
        |-- test
            |-- data
            +-- images
            +-- lidar
            +-- maps
            +-- v1.0-test       <-- symlink to `data`, see command below to create this
        |-- train
            |-- data
            +-- images
            +-- lidar
            +-- maps
            +-- v1.0-trainval   <-- symlink to `data`, see command below to create this
        +-- sample_submission.csv
        +-- train.csv
```
once you have all the directories other than the symlinks in place, issue these commands to create the symlinks:
```
cd ~/LyftObjDetDataset/test
ln -s data v1.0-test

cd ~/LyftObjDetDataset/train
ln -s data v1.0-trainval
```
make test `instance.json` and `sample_annotation.json` files with a blank python list `[]`
```
cd ~/LyftObjDetDataset/test/data
touch instance.json
gedit instance.json
# type a blank python list, [], then save
touch sample_annotation.json
gedit sample_annotation.json
# type a blank python list, [], then save
```

#### 4) clone and setup this repository

make the directory `~/workspace-av` if it does not exist already
```
cd ~
mkdir workspace-av
```
clone this repository to `~workspace-av`
```
cd ~/workspace-av
git clone https://github.com/cdahms123/lyft-3d-det.git
```

#### 5) If using a GPU with < 24GB, reduce training batch size

If you're using a GPU with < 24GB, you will need to open the open the config file:
```
gedit ~/workspace-av/lyft-3d-det/configs/settings.config
```
Then scroll towards the end and reduce the training batch size from 4 to 2:
```
.
.
.
train_input_reader: {
  .
  .
  .  
  batch_size: 4    <= reduce this to 2 if using a < 24 GB GPU
  preprocess: {
    .
    .
    .
  }
}
.
.
.
```

#### 6) run

Run the scripts `0` through `4` in the root of this repository in order, here is a quick description of what each does:<br>

`0_lyft_intro.py` - a quick tour of using the Lyft Object Detection Dataset<br>
<br>
`1_visualize_dataset.py` - visualizes a frame in the training portion of the dataset with Plotly, only ground truth boxes are shown since training hasn't been performed yet<br>
<br>
`2_train.py` - splits the train data into train/val (the split is done dynamically while the program runs and does not alter the dataset in any way), runs training process, inferences on test set, saves graph, val results, and test results to `/results`<br>
<br>
`3_visualize_val.py` - visualizes frames that ended up in the validation split with Plotly, both predicted boxes and ground truth boxes are shown<br>
<br>
`4_visualize_test.py` - visualizes a frame in the test portion of the dataset, since there are no ground truth boxes for the test set only the predicted boxes are shown<br>
<br>
**Note:** to re-train, delete the directory `results/` before running `2_train.py` again, if `results/` exists when `2_train.py` is started you will get an error

#### 7) upload submission csv file

After `2_train.py` has been ran, in `results/` there will be a file `kaggle_sub.csv`, to make a late submission to the Kaggle competition upload `kaggle_sub.csv` to https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/submit












