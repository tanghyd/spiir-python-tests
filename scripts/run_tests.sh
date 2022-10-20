#!/bin/bash
#
#SBATCH --job-name=test_zerolags
#SBATCH --output=logs/test_zerolags_%j.log
#SBATCH --error=logs/test_zerolags_%j.log
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=12g
#SBATCH --cpus-per-task=4
#SBATCH --requeue

mkdir -p logs/

# test_dir=/fred/oz996/tdavies/spiir_project/sources/testing/gout/py3/MR62_tests
test_dir=/fred/oz996/tdavies/spiir_project/sources/testing/gout/py3/MR70_tests
a=${test_dir}/run1/000/000_zerolag_1187008582_334.xml.gz
b=${test_dir}/run2/000/000_zerolag_1187008582_334.xml.gz


source /fred/oz016/dtang/envs/spiir-data-py310/venv/bin/activate

srun python tests/test_zerolags.py ${a} ${b} --verbose