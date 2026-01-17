This project is the code for the paper, "A lightweight road crack detection model based on multilevel thresholding image segmentation", submitted to Computer-Aided Civil and Infrastructure Engineering for consideration. The test datasets are included. The proposed model is ligtweight and effective.

## Requirements

All dependencies for this project are listed in the `environment.yml` file.

## Usage

The model is trained in three stages. Follow the steps below in order. 

**1. Preprocess Data** Run `crack_self_test.py` to prepare your dataset.

**2. Pre-train Encoder** Use `train2.py` to pre-train the encoder network. The best weights will be saved in the `checkpoints` directory. 

**3. Train Decoder** To train the decoder, you must first load the pre-trained encoder weights. 

**Modify** `decoder_train5.py`: Open the script and update the path variable for the encoder weights to point to the file generated in the previous step. * **Run the training**:  The final model will be saved to the `decoder_checkpoints` directory.

**4. Complexity and inference speed testing** Use the `test_fps_flops.py` script.  You must load the encoder and decoder weights and specify the image.

**5.Generalization testing** Run the `cross_dataset_generalization_test.py` script.

**6.Comparison of Multi-level Threshold Image Segmentation Methods**  Use the `multi_level_thresholding_segmentation_evalution_test_2.py` script, changing different methods and histogram type parameters to generate data.Then use the`multi_level_thresholding_segmentation_evalution_test_3.py` script to generate visualization data.

**7.File export for edge devices.** Use the `export_neural_eris_system_trt_fix.py` file to export the model as an ONNX file for subsequent use on edge devices.

**Trained weight files**We provide the pre-trained encoder weight file 48.pth, the pre-trained decoder weight file decoder_epoch_187.pth, and the mtscrack_neural_eris_static.onnx file exported using export_neural_eris_system_trt_fix.py, all ready for direct use.