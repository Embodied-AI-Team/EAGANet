python -m pdb visu_action_heatmap_proposals_custom_3f_multi.py \
  --exp_name exp-model_3d-grasp-None-train_3d_robotiq_3f \
  --model_epoch 184 \
  --model_version model_3d \
  --overwrite

: << 'COMMENT'
-- robotiq expriment --
python -m pdb visu_action_heatmap_proposals_robotiq.py \
  --exp_name exp-model_3d-grasp-None-train_3d_pandas \
  --model_epoch 256 \

-- kinova expriment --
python -m pdb visu_action_heatmap_proposals_kinova_single.py \
  --exp_name exp-model_3d-grasp-None-train_3d_kinova_3f \
  --model_epoch 100 \

-- allegro expriment --
yellow_cup:   --model_epoch 880
JarroDophilusFOS_Value_Size:   --model_epoch 100

python -m pdb visu_action_heatmap_proposals_allegro_4f.py \
  --exp_name exp-model_3d-grasp-None-train_3d_kinova_3f \
  --model_epoch 100 \

-- hand expriment --
python -m pdb visu_action_heatmap_proposals_hand.py \
  --exp_name exp-model_3d-grasp-None-train_3d_kinova_3f \
  --model_epoch 100 \


-- mutil real expriment --
python -m pdb visu_action_heatmap_proposals_custom_3f_multi.py \
  --exp_name exp-model_3d-grasp-None-train_3d_robotiq_3f \
  --model_epoch 184 \
  --model_version model_3d \
  --overwrite
COMMENT