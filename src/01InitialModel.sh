#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=1:mem=200gb
#PBS -N InitialModel

cd /rds/general/user/ms7024/home/kaggle_RNAfold/src

module load anaconda3/personal
source activate r413

python 01InitialModel.py