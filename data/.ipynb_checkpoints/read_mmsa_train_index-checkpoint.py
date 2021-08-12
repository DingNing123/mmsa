import os
import numpy as np
import pandas as pd

label_dir = "/media/dn/85E803050C839C68/datasets/CH-SIMS/metadata"
train_index = np.array(pd.read_csv(os.path.join(label_dir, 'train_index.csv'))).reshape(-1)
print(type(train_index))

data = {'index_dn':train_index}
data_df = pd.DataFrame(data)

data_df.to_csv(os.path.join(label_dir, 'dn1.csv'),index=False)