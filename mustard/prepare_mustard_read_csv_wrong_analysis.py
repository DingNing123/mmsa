# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('false_predict.csv')

sheldon_true,sheldon_false,howard_true,howard_false = [],[],[],[]
# print(df.to_string())
print(len(df),df.__class__)
for i in range(len(df)):
    row = df.iloc[i]
    if row['right_label']==False and row['speaker']=='SHELDON':
        sheldon_false.append(row['segments'])

    if row['right_label']==True and row['speaker']=='SHELDON':
        sheldon_true.append(row['segments'])

    if row['right_label'] == False and row['speaker'] == 'HOWARD':
        howard_false.append(row['segments'])

    if row['right_label'] == True and row['speaker'] == 'HOWARD':
        howard_true.append(row['segments'])

print(len(sheldon_false),len(sheldon_true),len(howard_false),len(howard_true))# 19 14 8 8
print("total:", len(sheldon_false)+len(sheldon_true)+len(howard_false)+len(howard_true))

name_list = ['SHELDON', 'HOWARD']
num_non_sarcastic = [len(sheldon_false),len(howard_false) ] # 19 8
num_sarcastic = [len(sheldon_true), len(howard_true)] # 14 8
plt.figure(1, figsize=(9, 6))
plt.title("wrong predict",fontsize=20)
plt.bar(range(len(num_non_sarcastic)), num_non_sarcastic, width=0.5, label='non_sarcastic')
plt.bar(range(len(num_non_sarcastic)), num_sarcastic, width=0.5, bottom=num_non_sarcastic, label='sarcastic', tick_label=name_list)
plt.legend(fontsize=20)
plt.ylabel('number',fontsize=20)
plt.xlabel('speaker',fontsize=20)
# plt.show()
output_filename = "wrong_predict.png"
plt.savefig(output_filename, bbox_inches='tight', dpi=300)
plt.close()
print(f"write to {output_filename}!")

