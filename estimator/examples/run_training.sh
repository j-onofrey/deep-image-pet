SemanticClassifier.py \
  --train \
  -d \
  -i mni_training.txt \
  -t masks_training.txt \
  -o trained_model/ \
  -loss CrossEntropy \
  -model UNet2d \
    --dropout_rate 0.15 \
    --num_output_channels 2 \
  --patch_size 128 128 \
  --num_epochs 2 \
  --epoch_size 1600 \
  --batch_size 16 \
  --augment \
    --flip 0 \
    