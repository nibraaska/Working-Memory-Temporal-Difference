import glob
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

max_targets = []

for filename in glob.glob('logs/*'):
    max_target = 0
    with open(filename,'r') as f:
        all_logs = f.readlines()
        for log in all_logs:
            target = json.loads(log)['target']
            if target > max_target:
                max_target = target
    max_targets += [max_target]

sns.kdeplot(max_targets, label="Static", shade=True)
