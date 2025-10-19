This project is the code for the paper, "A lightweight road crack detection model based on multilevel thresholding image segmentation", submitted to Computer-Aided Civil and Infrastructure Engineering for consideration. The test datasets are included. The proposed model is ligtweight and effective.

## Requirements

All dependencies for this project are listed in the `environment.yml` file.

## Usage

The model is trained in three stages. Follow the steps below in order. 

**1. Preprocess Data** Run `crack_self_test.py` to prepare your dataset.

**2. Pre-train Encoder** Use `train2.py` to pre-train the encoder network. The best weights will be saved in the `checkpoints` directory. 

**3. Train Decoder** To train the decoder, you must first load the pre-trained encoder weights. 

**Modify** `decoder_train5.py`: Open the script and update the path variable for the encoder weights to point to the file generated in the previous step. * **Run the training**:  The final model will be saved to the `decoder_checkpoints` directory.
