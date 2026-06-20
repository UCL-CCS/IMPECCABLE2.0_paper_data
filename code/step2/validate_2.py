import json
import pandas as pd
json_file = 'config_training.json'

try:
    with open(json_file) as f:
        hyper_params = json.load(f)
    
    df = pd.read_csv(hyper_params['callbacks']['log_csv'])
    if max(df['val_r2']) < 0.2:
        print("Not OK")
    else:
        print("OK")
except:
    print("Not OK")
