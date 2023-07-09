# GLORY
Code for GLORY

##### Enviroment
> Python 3.8.10
> pytorch 1.13.1+cu117
```shell
cd GLORY

apt install unzip python3.8-venv python3-pip
python3 -m venv venv
source venv/bin/activate

# pip install
pip3 install torch==1.13.0 
pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip3 install transformers
pip3 install numpy pandas nltk scikit-learn
pip3 install pyrootutils tqdm swifter
pip3 install omegaconf wandb hydra-core optuna hydra-optuna-sweeper
```

```shell
# wandb setting
export WANDB_MODE=offline

# dataset download
cd data
bash ../scripts/data_download.sh

# Test
python3 src/main.py model=GlobalLocal dataset=MINDsmall 
```
