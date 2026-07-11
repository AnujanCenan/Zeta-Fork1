import pandas as pd
import ast
from dotenv import load_dotenv
import os

load_dotenv()
PTBXL_PATH = os.getenv("PTBXL_DATASET")


db = pd.read_csv(os.path.join(PTBXL_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
for eid, row in db.iterrows():
    codes = ast.literal_eval(row['scp_codes'])
    if '1AVB' in codes and codes['1AVB'] == 100.0:
        print(eid, codes)
        break