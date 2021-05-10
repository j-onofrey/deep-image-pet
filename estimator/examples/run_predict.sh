SemanticClassifier.py \
  -d \
  -i mni_testing.txt \
  -o predict/ \
  -model_path trained_model/ \
  -model UNet2d \
    --dropout_rate 0.0 \
    --num_output_channels 2 \
  --patch_size 128 128 \
  --batch_size 16 \
  