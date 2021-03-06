#!/bin/bash
#SBATCH --time=0-0:5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=
#SBATCH --mail-type=ALL


#H="/home/masearaujo/"
H="/home/masearaujo"

#export LD_PRELOAD=/home/masearaujo/glibc/lib/libc-2.18.so
export PATH=/home/masearaujo/glibc/bin:$PATH
export PATH=/home/masearaujo/glibc/sbin:$PATH
export LIBRARY_PATH=/home/masearaujo/glibc/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/masearaujo/glibc/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/masearaujo/glibc/lib/libc-2.18.so
export CPATH=/home/masearaujo/glibc/include:$CPATH
export C_INCLUDE_PATH=/home/masearaujo/glibc/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/masearaujo/glibc/include:$CPLUS_INCLUDE_PATH


source "$H/Envs/PyEnv/scii-teste1/bin/activate"
python "$H/Git/URNAI-Tools/urnai/solve_simple64.py"
