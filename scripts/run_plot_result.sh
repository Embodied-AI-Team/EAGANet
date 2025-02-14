python -m pdb plot_result_hand.py \
  --src_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_pandas \
  --record_names potato_chip_1_1040 potato_chip_1_1041 potato_chip_1_1042 potato_chip_1_1043 potato_chip_1_1044 potato_chip_1_1456 potato_chip_1_1395 potato_chip_1_1436 \
  --random_seed 20

: << 'COMMENT'
== robotiq expriment ==
  --src_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_pandas \
  --record_names yellow_cup_1019 yellow_cup_1221 yellow_cup_1236 yellow_cup_1319 yellow_cup_1378 yellow_cup_1482 yellow_cup_1539 yellow_cup_1549 \
  --random_seed 20

  --src_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_pandas \
  --record_names potato_chip_1_1045 potato_chip_1_1046 potato_chip_1_1047 potato_chip_1_1048 potato_chip_1_1210 \
  --random_seed 20

== custom_3f expriment ==
  --src_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_pandas \
  --record_names  yellow_cup_1018 yellow_cup_1019 yellow_cup_1549 yellow_cup_1550 yellow_cup_1624 yellow_cup_1630 \
  --random_seed 20

  --src_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_pandas \
  --record_names potato_chip_1_1045 potato_chip_1_1046 potato_chip_1_1047 potato_chip_1_1300 potato_chip_1_1358 \
  --random_seed 20

== allegro_4f expriment ==
  --src_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_pandas \
  --record_names  yellow_cup_1018 yellow_cup_1019 yellow_cup_1549 yellow_cup_1550 yellow_cup_1624 yellow_cup_1630 \
  --random_seed 20

  --src_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_pandas \
  --record_names potato_chip_1_1045 potato_chip_1_1046 potato_chip_1_1047 potato_chip_1_1300 potato_chip_1_1358 \
  --random_seed 20
== hand expriment ==
python -m pdb plot_result_hand.py \
  --src_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_inspired_hand \
  --record_names  yellow_cup_1039 yellow_cup_1040 yellow_cup_1102 yellow_cup_1118 yellow_cup_1119 yellow_cup_1120 \
  --random_seed 20

python -m pdb plot_result_hand.py \
  --src_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_pandas \
  --record_names potato_chip_1_1042 potato_chip_1_1043 potato_chip_1_1044 potato_chip_1_1456 potato_chip_1_1395 potato_chip_1_1436 \
  --random_seed 20
COMMENT