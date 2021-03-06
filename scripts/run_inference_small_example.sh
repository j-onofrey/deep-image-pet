estimator/SemanticRegressor.py \
  --device gpu:0 \
  -d \
  -i /data7/mct/Yihuan/For_Takuya/Oncology/Sub_list/training_input_FDGWBtransfer_linux_N40 \
  -o inference \
  -model_path results \
  -model UNet3d_generalized_nomaxpool \
    --num_output_channels 1 \
    --filter_size 3 \
    --num_levels 2 \
    --dilation_rate 1 \
    --num_convs_per_level 2 \
    --num_filters 32 \
    --bridge_mode concat \
    --normalization batch \
  --alpha_GDL 1.0 \
  --patch_size 64 64 32 \
  --pad_size 0 0 8 \
  --pad_type edge \
  --batch_size 16
