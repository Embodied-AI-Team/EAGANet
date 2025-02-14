import os

# 设置你想要读取子文件夹的父文件夹路径
parent_folder_path = '/media/zhou/软件/博士/具身/grasp_train_data_gspwth_kinova_3f'

# 获取所有子文件夹
sub_folders = [name for name in os.listdir(parent_folder_path)
               if os.path.isdir(os.path.join(parent_folder_path, name))]

# 将子文件夹名称写入txt文件
with open('/media/zhou/软件/博士/具身/data_tuple_list.txt', 'w') as file:
    for folder in sub_folders:
        file.write(folder + '\n')

print('子文件夹名称已写入sub_folders.txt文件中。')