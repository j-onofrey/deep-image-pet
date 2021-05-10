SemanticGenerator.py \
  -d \
  --device gpu:1 \
  -i sampledata/3dbrain_mr_test.txt \
  -o predict/ \
  -model_path trained_model/ \
  --smoothing_sigma 0.1 \
  --stride_size 8 8 8 \
  -model ConditionalGAN \
    --model_generator UNet3d_generalized \
      --num_output_channels_gen 1 \
      --dropout_rate_gen 0.0 \
      --num_levels_gen 2 \
      --num_convs_per_level_gen 3 \
      --num_filters_gen 32 \
  --patch_size 32 32 32 \
  --batch_size 64 \
