# SPIIR Python Tests

This is a work in progress repository for testing SPIIR data artifacts on the OzStar
supercomputing cluster.

## Quick Start

To run the example test to compare columns between two SPIIR zerolag output files
(LIGOLW XML files with SPIIR's custom PostcohInspiralTable schema), we can simply run
the run_tests.sh script in scripts/.

    sbatch scripts/run_tests.sh

This will submit the tests job to the cluster via SLURM, based on the modules specified
in the tests/ directory (i.e. test_zerolags.py), and output the results to the logs/
directory.