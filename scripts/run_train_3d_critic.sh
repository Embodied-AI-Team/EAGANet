python -m pdb train_3d_critic.py \
    --exp_suffix train_3d_critic_pandas \
    --model_version model_3d_critic \
    --primact_type grasp \
    --data_dir_prefix datasets/gt_data \
    --offline_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_kinova_3f\
    --val_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_kinova_3f \
    --val_data_fn data_tuple_list.txt \
    --train_shape_fn datasets/grasp_stats/train_data_list.txt \
    --buffer_max_num 200 \
    --num_processes_for_datagen 4 \
    --num_interaction_data_offline 50 \
    --num_interaction_data 1 \
    --resume

