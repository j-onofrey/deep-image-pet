estimator/SemanticRegressor.py \
  --train \
  --device gpu:0 \
  -d \
  -i /data7/mct/Yihuan/For_Takuya/Oncology/Sub_list/training_input_FDGWBtransfer_linux_N40 \
  -t /data7/mct/Yihuan/For_Takuya/Oncology/Sub_list/training_target_FDGWBtransfer_linux_N40 \
  -o results \
  -loss L1 \
  -model UNet3d_generalized_nomaxpool \
    --opt_learning_rate 0.001 \
    --opt_learning_decay_rate 0.9747 \
    --dropout_rate 0.15 \
    --num_output_channels 1 \
    --filter_size 3 \
    --num_levels 4 \
    --dilation_rate 1 \
    --num_convs_per_level 2 \
    --num_filters 64 \
    --bridge_mode concat \
    --normalization batch \
  --alpha_GDL 1.0 \
  --patch_size 64 64 32 \
  --pad_size 0 0 8 \
  --pad_type edge \
  --num_epochs 160 \
  --epoch_size 10000 \
  --batch_size 16
