import pandas as pd

try:
    df = pd.read_csv('sorted_data.csv')
    if len(df)<8500:
        print("Not OK")
    else:
        print("OK")

except:
    print("Not OK")
