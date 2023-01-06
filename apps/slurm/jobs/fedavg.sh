#!/bin/bash
#SBATCH --time=1-23:00:00
#SBATCH --account=def-zdziong
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=mohamad-arfah.dabberni.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL
#SBATCH --output=job_basic.out
HOME=/home/ahmmmoud/projects/def-zdziong/ahmmmoud/arafeh/geneticfed


module load python/3.6
source ${HOME}/venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${HOME}"

cd results || python ../fedavg.py -e 25 -b 80000 -r 500 -cr 5 -lr 0.01 -t 'fedavg'
