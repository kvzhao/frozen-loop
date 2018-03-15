import json
import os
from pprint import pprint

ofile = 'log.json'

sites = [123, 124, 223, 224]

d = {
    'traj': sites,
    'start_point': [123]
}

def append_record(record):
    with open(ofile, 'a') as f:
        json.dump(record, f)
        f.write(os.linesep)

for i in range(10):
    my_dict = {'number':i, 'traj': sites}
    append_record(my_dict)