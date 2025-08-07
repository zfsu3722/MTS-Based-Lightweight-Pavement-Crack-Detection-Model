# Code for [ A Lightweight Road Crack Detection Model Based on Multilevel Thresholding Image Segmentation ]

![image-20250807130716029](C:\Users\29657\AppData\Roaming\Typora\typora-user-images\image-20250807130716029.png)

Overall architecture of the proposed hierarchical crack detection framework. The process starts with the ERIS module decomposing the input image into K binary planes. These planes are processed by a shared-weight encoder. The resulting features are concatenated and fed into a lightweight decoder, which reconstructs the final segmentation mask. The entire network is trained end-to-end under the supervision of a combined loss function

## Requirements

All dependencies for this project are listed in the `environment.yml` file.

## Usage

The model is trained in three stages. Follow the steps below in order. 

**1. Preprocess Data** Run `crack_self_test.py` to prepare your dataset.

**2. Pre-train Encoder** Use `train2.py` to pre-train the encoder network. The best weights will be saved in the `checkpoints` directory. 

**3. Train Decoder** To train the decoder, you must first load the pre-trained encoder weights. 

**Modify** `decoder_train5.py`: Open the script and update the path variable for the encoder weights to point to the file generated in the previous step. * **Run the training**:  The final model will be saved to the `decoder_checkpoints` directory.

## Citation

If you use this code in your research, please consider citing our paper:

####################################################

## Contact

For any questions, please contact ###############