import os
import os.path as osp
import glob
# import csv
import pandas as pd
for save_dir in os.listdir("results"):
    # exps = os.listdir(osp.join("results", save_dir))
    runs = sorted(glob.glob(osp.join("results", save_dir,  'experiment_*')))
    run_id = int(runs[-1].split('_')[-1])
    print(save_dir)
    with open(osp.join("results", save_dir, "experiment_"+str(run_id), "log.csv")) as f_csv:
        # reader = csv.DictReader(f_csv, delimiter=',')
        reader = pd.read_csv(f_csv)
        print(reader['mse'].mean())
        filter = ([item for item in reader['mse'] if item < 5])
        if len(filter)> 0:
            print(sum(filter)/len(filter))

