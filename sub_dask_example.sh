#!/bin/sh
#SBATCH -D ./
#SBATCH -J example
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#SBATCH -p nocona
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 48:00:00

py_env=/home/rmanser/software/mambaforge/envs/dask_example/bin/python

init=$1
n_jobs=${2:-''}

time $py_env dask_example.py $init --n_jobs $n_jobs