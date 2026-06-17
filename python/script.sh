#!/usr/bin/env bash

set +e

python3 run_exp.py -r 128 -i exp_res_128
python gen_latex_table.py -e ../data/synthetic-1/results/exp_res_128

python3 run_exp.py -r 256 -i exp_res_256
python gen_latex_table.py -e ../data/synthetic-1/results/exp_res_256

python3 run_exp.py -r 512 -i exp_res_512
python gen_latex_table.py -e ../data/synthetic-1/results/exp_res_512

python3 run_exp_cross.py -r 512 -i exp_cross
