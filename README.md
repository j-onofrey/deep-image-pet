# DeepImage for PET Imaging

DeepImage-Pet is a software framework for training and running deep neural networks using TensorFlow. This code is part of the [BioImage Suite](https://github.com/bioimagesuiteweb) family of software tools for biomedical image analysis. This framework can be used to perform either segmentation (classification) or regression in both 2D and 3D imaging data.

## Tutorial
Assuming you have cloned the DeepImage repository, you can find example scripts referenced in `deep-image-pet/estimator/examples/`.

Here, we provide an example to train a deep neural network using the U-Net model for image classification. In this example, we will segment the prostate from 3D magnetic resonance (MR) imaging (MRI). From a small dataset of 3 de-identified subjects, we select 2 subjects for training and 1 subject for testing. These images can be found in `deep-image-pet/data/sample_data`.

### Setup

Install the [anaconda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) for deep-image-pet using the provided `environment.yml` file.
```
$ conda env create -f environment.yml
```
Once setup, you can activate your conda environment:
```
$ conda activate deep-image-pet
```

For convenience, you can add the `deep-image-pet/estimator/` folder to your PATH by sourcing the following script on the command line (using the BASH shell):

```
$ source [INSTALL_PATH]/deep-image-pet/estimator/setpaths.sh
```
You will now be able to access the driver program without have to use the entire `INSTALL_PATH`.

### Data formatting
DeepImage takes input as a list of training images in a text file. For training, you will need two lists: i) training input data, and ii) target training data. For the sample data example, the input data is listed in `deep-image-pet/estimator/examples/sample_data_input_training.txt`:

```
../../data/sample_data/prostate_T2_1.nii.gz
../../data/sample_data/prostate_T2_2.nii.gz
```

And the target training data is listed in `deep-image-pet/estimator/examples/sample_data_target_training.txt`:

```
../../data/sample_data/prostate_segm_1.nii.gz
../../data/sample_data/prostate_segm_2.nii.gz
```

**Note:** The files in the input and target data files must correspond to each other, e.g. subject 1's image data and target data are both on the first line of each file. Unequal number of files in each file will result in an error, and mismatched data will result in the model learning something non-sensical.

Similarly, the testing images are list in the file `deep-image-pet/estimator/examples/sample_data_input_testing.txt`:

```
../../data/sample_data/prostate_T2_3.nii.gz
```

**TIP:** For your code, it is recommended that you prefix filenames with the absolute system path.

## Training a model

The type of problem that you want to run determines which DeepImage model framework to use. For this example, we will be performing semantic segmentation, which is a form of patch-based image segmentation, and we use the SemanticClassifier program. On the terminal, you can see the various options available for this particular model by typing:
```
$ SemanticClassifier.py -h 
```
which will list the various program options:
```
usage: SemanticClassifier.py [-h] -i INPUT_DATA -model MODEL -model_path
                             MODEL_PATH -o OUTPUT_PATH
                             [--stride_size STRIDE_SIZE [STRIDE_SIZE ...]]
                             [--smoothing_sigma SMOOTHING_SIGMA]
                             [-b BATCH_SIZE] [-p PATCH_SIZE [PATCH_SIZE ...]]
                             [--target_patch_offset TARGET_PATCH_OFFSET [TARGET_PATCH_OFFSET ...]]
                             [--unpaired] [--pad_size PAD_SIZE [PAD_SIZE ...]]
                             [--pad_type PAD_TYPE] [-d] [--device DEVICE]
                             [--fixed_random] [--one_hot_output]

Recon: Semantic classifier driver class Use the --train flag to switch to
training mode. (Use --train -h to see training help)

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DATA, --input_data INPUT_DATA
                        txt file containing list of input files
  -model MODEL          Model name (can be found in the model module)
  -model_path MODEL_PATH
                        Path to the saved model
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save the prediction output
  --stride_size STRIDE_SIZE [STRIDE_SIZE ...]
                        List of stride lengths for patches. If no value is
                        given for a particular index, the stride size is set
                        to 1. Default value is None.
  --smoothing_sigma SMOOTHING_SIGMA
                        Sigma parameter for smoothing of predicted image
                        reconstructed from patches, default = 0.0
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Num of training samples per mini-batch
  -p PATCH_SIZE [PATCH_SIZE ...], --patch_size PATCH_SIZE [PATCH_SIZE ...]
                        patch size (should be even). If no value is given, the
                        patch size is the same as the input data. If more than
                        one value is given the patch will be set according to
                        inputs. Default value is None.
  --target_patch_offset TARGET_PATCH_OFFSET [TARGET_PATCH_OFFSET ...]
                        target patch offset. Default value is None.
  --unpaired            Specify that the input data files are not paired,
                        default=False
  --pad_size PAD_SIZE [PAD_SIZE ...]
                        data padding size. If more than one value is given the
                        data will be padded according to input values. Default
                        value is None, no data padding will be used for the
                        input data.
  --pad_type PAD_TYPE   Type of data padding to perform, can be one of: edge,
                        reflect, or zero. Default zero
  -d, --debug           Set debug output level, default=False (info level)
  --device DEVICE       Train using device, can be cpu:0, gpu:0, gpu:1, if
                        available, default gpu:0
  --fixed_random        If true, the random seed is set explitily (for
                        regression testing)
  --one_hot_output      Flag to use one hot output format
```
For training, the most important flag is the --train flag, which enables all the additional training command line parameters:
```
$ SemanticClassifier.py --train -h
```
Which now displays options for model loss functions and optimizers:
```
usage: SemanticClassifier.py [-h] [--train] -i INPUT_DATA -model MODEL -t
                             TARGET_DATA
                             [--target_patch_size TARGET_PATCH_SIZE [TARGET_PATCH_SIZE ...]]
                             -o OUTPUT_PATH
                             [--validation_input_data VALIDATION_INPUT_DATA]
                             [--validation_target_data VALIDATION_TARGET_DATA]
                             [--validation_patch_size VALIDATION_PATCH_SIZE [VALIDATION_PATCH_SIZE ...]]
                             [--validation_metrics VALIDATION_METRICS [VALIDATION_METRICS ...]]
                             [--max_iterations MAX_ITERATIONS]
                             [--epoch_size EPOCH_SIZE]
                             [--num_epochs NUM_EPOCHS] [--augment]
                             [--param_file PARAM_FILE] [-b BATCH_SIZE]
                             [-p PATCH_SIZE [PATCH_SIZE ...]]
                             [--target_patch_offset TARGET_PATCH_OFFSET [TARGET_PATCH_OFFSET ...]]
                             [--unpaired] [--pad_size PAD_SIZE [PAD_SIZE ...]]
                             [--pad_type PAD_TYPE] [-d] [--device DEVICE]
                             [--fixed_random]

Train: Semantic classifier driver class

optional arguments:
  -h, --help            show this help message and exit
  --train               force training mode
  -i INPUT_DATA, --input_data INPUT_DATA
                        txt file containing list of input files
  -model MODEL          Model name (can be found in the model module)
  -t TARGET_DATA, --target_data TARGET_DATA
                        txt file containing list of training target files
  --target_patch_size TARGET_PATCH_SIZE [TARGET_PATCH_SIZE ...]
                        target patch size. If no value is given, the target
                        patch size is the same as the patch_size. If more than
                        one value is given the patch will be set according to
                        inputs. Default value is None.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save the learned model to
  --validation_input_data VALIDATION_INPUT_DATA
                        txt file containing list of validation input files
  --validation_target_data VALIDATION_TARGET_DATA
                        txt file containing list of validation target files
  --validation_patch_size VALIDATION_PATCH_SIZE [VALIDATION_PATCH_SIZE ...]
                        patch size (should be even). If no value is given, the
                        patch size is the same as the input data. If more than
                        one value is given the patch will be set according to
                        inputs. Default value is None.
  --validation_metrics VALIDATION_METRICS [VALIDATION_METRICS ...]
                        List of evaluation metrics to use, can be one or more
                        of ['accuracy', 'dice', 'precision', 'recall', 'rmse',
                        'mae', 'psnr']. Default value is None.
  --max_iterations MAX_ITERATIONS
                        Maximum number of training iterations. If None, then
                        training will run for till all epochs complete.
  --epoch_size EPOCH_SIZE
                        Number of trianing samples per epoch.
  --num_epochs NUM_EPOCHS
                        Maximum number of training iterations.
  --augment             Flag to include data augmentation
  --param_file PARAM_FILE
                        JSON File to read parameters from, same format as
                        output .json
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Num of training samples per mini-batch
  -p PATCH_SIZE [PATCH_SIZE ...], --patch_size PATCH_SIZE [PATCH_SIZE ...]
                        patch size (should be even). If no value is given, the
                        patch size is the same as the input data. If more than
                        one value is given the patch will be set according to
                        inputs. Default value is None.
  --target_patch_offset TARGET_PATCH_OFFSET [TARGET_PATCH_OFFSET ...]
                        target patch offset. Default value is None.
  --unpaired            Specify that the input data files are not paired,
                        default=False
  --pad_size PAD_SIZE [PAD_SIZE ...]
                        data padding size. If more than one value is given the
                        data will be padded according to input values. Default
                        value is None, no data padding will be used for the
                        input data.
  --pad_type PAD_TYPE   Type of data padding to perform, can be one of: edge,
                        reflect, or zero. Default zero
  -d, --debug           Set debug output level, default=False (info level)
  --device DEVICE       Train using device, can be cpu:0, gpu:0, gpu:1, if
                        available, default gpu:0
  --fixed_random        If true, the random seed is set explitily (for
                        regression testing)
```

You can also access specific model parameters after you have specified the model you want to use. Here, using the 2D U-Net model model/UNet2dpy, you can see the various model-specific parameters:
```
$ SemanticClassifier.py --train -model UNet2d -h
```
which will display all available parameters including the U-Net's:
```
...
U-Net 2D model params:
  --num_output_channels NUM_OUTPUT_CHANNELS
                        Number of output channels in the final layer, default
                        2 (for binary classification)
  --dropout_rate DROPOUT_RATE
                        Dropout rate, default = 0.0
  --filter_size FILTER_SIZE
                        Filter size in convolutional layers
  --num_filters NUM_FILTERS
                        Number of convolution filters in the first layer
  --num_levels NUM_LEVELS
                        Number of U-Net levels (essentailly the number of max
                        pools
  --num_convs_per_level NUM_CONVS_PER_LEVEL
                        Number of convolutions per U-Net level
  --bridge_mode BRIDGE_MODE
                        Bridge mode (one of concat, none)
...
```
A training script `deep-image-pet/estimator/examples/sample_training.sh` is provided that will train the model with the following parameters:
```
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
```
For each training epoch (a total of 10 specified here), this script tells the driver to extract 160 128x128x1 2D patches from the 3D image volume at random (with replacement) and then feeds them to the 2D U-Net model for training in mini-batch sizes of 16 at a time. For this particular problem, we include the option to augment the training data using the `--augment` flag and then specify using only flips along the 0-th axis (the x-axis) using the flag `--flip 0`. This model will be saved into the newly created `sample_model/` directory.

**Note:** The `--num_output_channels` parameter is set to 2 here because this is a binary segmentation problem, i.e. we are segmenting two classes: the background and foreground (the prostate gland). This number can be changed for multi-class segmentation problems. It can also be set to 1 for regression problems like image synthesis or image denoising. For these problems, you will want to use the `SemanticRegressor` framework instead of `SemanticClassifier`.

To run the script on the command line:
```
$ sh sample_training.sh
```
You can view the model and the training progress using TensorBoard:
```
$ tensorboard --logdir=trained_model/ --port 8888
```
which will start up a web server that can be viewed in your web browser by going to `http://localhost:8888`

## Testing the model
To use the model to segment test data (not included in the training), we create a and run a testing script `deep-image-pet/estimator/examples/sample_testing.sh`:

```
SemanticClassifier.py \
  -d \
  -i sample_data_input_testing.txt \
  -o sample_results/ \
  -model_path sample_model/ \
  -model UNet2d \
    --dropout_rate 0.0 \
    --num_output_channels 2 \
    --num_filters 16 \
    --num_levels 3 \
  --patch_size 128 128 1 \
  --batch_size 16 \
```
The test (inference) script requires the `-model_path MODEL_PATH` parameter, which indicates where the saved model weights were saved during the training process. This value will correspond to the `-o MODEL_PATH` in the training script. You also must specify the model parameters as done in the training script since there is no way to recreate the model architecture from the saved model. Note, that some model parameters can be changed at testing time, e.g. `--dropout_rate`, while most other must stay the same, e.g. `num_filters`, etc. 

Run the script:

```
$ sh sample_testing.sh 
```
The output segmentation results will appear in the folder `deep-image-pet/estimator/examples/sample_results/`.

## Regression problem
In our recent work, we proposed a novel imaging physics-based loss function, a line-integral projection (LIP-loss), to learn a CT-derived attenuation map (µ-CT) to derive deep learning-based attenuation map (µ-CT) from PET activity (λ-MLAA) and attenutaion maps (µ-MLAA) estimated by the maximum likelihood reconstruction of activity and attenuation (MLAA) algorithm.

A example **training** script is provided that will train the model with the following parameters:
```
SemanticRegressor.py \
  --train \
  --device gpu:0 \
  -d \
  -i train_2.5pct_G1_input.txt \
  -t train_2.5pct_G1_target.txt \
  -o model_FDG_40subj_pct2.5_inG1_tarG1_psz64x64x32_4Angles/ \
  -loss L1 \
  -model UNet3d_generalized_nomaxpool_LIL_multiAngle \
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
    --weight_LIL 0.2 \
    --num_angle 4 \
  --patch_size 64 64 32 \
  --pad_size 0 0 8 \
  --pad_type edge \
  --num_epochs 160 \
  --epoch_size 10000 \
  --batch_size 16
```

**Note:** `--weight_LIL WEIGHT_LIL` is the weight for LIP-loss and `--alpha_GDL WEIGHT_GDL` is the weight for GDL-loss (Please refer to the paper below for details). If the LIP-loss is applied, `--num_angle ANGLE_SIZE ` needs to be set for providing the number of LIP angles.

A example **testing** script is provided that will use the model to make predictions using test data (not included in the training) with the following parameters:
```
SemanticRegressor.py \
  -d \
  --device gpu:0 \
  -i predict_2.5pct_G1_input.txt \
  -o predicted_47subj_2.5pct_inG1_4Angles/ \
  -model_path model_FDG_40subj_pct2.5_inG1_tarG1_psz64x64x32_4Angles/ \
  --smoothing_sigma 0.25 \
  --stride_size 256 256 16 \
  -model UNet3d_generalized_nomaxpool_LIL_multiAngle \
    --dropout_rate 0.00 \
    --num_output_channels 1 \
    --filter_size 3 \
    --num_levels 4 \
    --dilation_rate 1 \
    --num_convs_per_level 2 \
    --num_filters 64 \
    --bridge_mode concat \
    --normalization batch \
  --patch_size 256 256 32 \
  --pad_size 0 0 8 \
  --pad_type edge \
  --batch_size 1
  
```

### Citation
If you use this code for your research or project, please cite:

    @article{toyonaga2022deep,
      title={Deep learning--based attenuation correction for whole-body PET—a multi-tracer study with 18F-FDG, 68 Ga-DOTATATE, and 18F-Fluciclovine},
      author={Toyonaga, Takuya and Shao, Dan and Shi, Luyao and Zhang, Jiazhen and Revilla, Enette Mae and Menard, David and Ankrah, Joseph and Hirata, Kenji and Chen, Ming-Kai and Onofrey, John A and others},
      journal={European Journal of Nuclear Medicine and Molecular Imaging},
      pages={1--12},
      year={2022},
      publisher={Springer}
      }
