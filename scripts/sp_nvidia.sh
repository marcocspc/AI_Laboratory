#!/bin/bash
#SBATCH --time=0-0:20
#SBATCH --partition=gpu
#SBATCH --gres:gpu:1
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

H="/home/masearaujo/"

source "$H/Envs/PyEnv/scii-teste1/bin/activate"
python "$H/Git/URNAI-Tools/urnai/solve_simple64.py"
