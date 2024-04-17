import os
# root path for the BraTS2018 dataset
data_path = '.'

f = open(os.path.join(data_path, 'name_mapping.csv'), 'w+')
f.writelines('Grade,BraTS_2018_subject_ID\n')
for label in ['HGG', 'LGG']:
    path = os.path.join(data_path, label)
    for dir_name in os.listdir(path):
        f.writelines(f'{label},{dir_name}\n')
f.close()
