SemanticClassifier.py \
  --train \
  -d \
  -i sample_data_input_training.txt \
  -t sample_data_target_training.txt \
  -o sample_model/ \
  -loss CrossEntropy \
  -model UNet2d \
    --dropout_rate 0.15 \
    --num_output_channels 2 \
    --num_filters 16 \
    --num_levels 3 \
  --patch_size 128 128 1 \
  --num_epochs 10 \
  --epoch_size 160 \
  --batch_size 16 \
  --augment \
    --flip 0 \
    
