#PBS -l walltime=30:00:00
#PBS -l select=1:ncpus=9:mem=400gb
#PBS -N PredDist

cd /rds/general/user/ms7024/home/kaggle_RNAfold/src

module load anaconda3/personal
source activate r413

python 02PredDist.py