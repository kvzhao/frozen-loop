import json

ofile = 'env_history.json'

with open(ofile, 'r') as f:
    my_list = [json.loads(line) for line in f]
    print (my_list)
    for d in my_list:
        print(d['Episode'])