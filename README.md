# Synthetic-Seismic-Velocity-Spectrum
This code is to generate the seismic supergather according to the assumed RMS velocity model. It also conclude the code for semblance spectrum computation.


## Preparation
- create conda env and install python packages
```shell
conda create -n SynData python=3.8
conda list -e > requirements.txt
```

## Generate synthetic velocity with constant SNR
There is a showcase generating a synthetic dataset with SNR=[10, 4, 2, 1, 2/3, 1/2, 2/5, 1/3].

Run the following codes in shell of Ubuntu system:
```shell
nohup python GenerateData.py --DataRoot /mnt/data/spectrum/hade --OutRoot /mnt/data/spectrum/hade-S1 --SNR 10 > Run.log 2>&1 &
nohup python GenerateData.py --DataRoot /mnt/data/spectrum/hade --OutRoot /mnt/data/spectrum/hade-S2 --SNR 4 > Run.log 2>&1 &
nohup python GenerateData.py --DataRoot /mnt/data/spectrum/hade --OutRoot /mnt/data/spectrum/hade-S3 --SNR 2 > Run.log 2>&1 &
nohup python GenerateData.py --DataRoot /mnt/data/spectrum/hade --OutRoot /mnt/data/spectrum/hade-S4 --SNR 1 > Run.log 2>&1 &
nohup python GenerateData.py --DataRoot /mnt/data/spectrum/hade --OutRoot /mnt/data/spectrum/hade-S5 --SNR 0.67 > Run.log 2>&1 &
nohup python GenerateData.py --DataRoot /mnt/data/spectrum/hade --OutRoot /mnt/data/spectrum/hade-S6 --SNR 0.5 > Run.log 2>&1 &
nohup python GenerateData.py --DataRoot /mnt/data/spectrum/hade --OutRoot /mnt/data/spectrum/hade-S7 --SNR 0.4 > Run.log 2>&1 &
nohup python GenerateData.py --DataRoot /mnt/data/spectrum/hade --OutRoot /mnt/data/spectrum/hade-S8 --SNR 0.33 > Run.log 2>&1 &
```

