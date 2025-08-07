import os
import numpy as np
import cv2
import time as tm

from crack500_support import aligned_img_gray_reconstruction

try:
    import crack500_support as cr5_spt
except ImportError as e:
    print(f"ERROR: Unable to import crack500_support.py. Please make sure it is in your Python path. {e}")
    exit()

train_dataset = "E:\\Road_Crack_Dataset\\Crack500_Dataset\\C5_seg_png_256_256\\train_img"
train_label ="E:\\Road_Crack_Dataset\\Crack500_Dataset\\C5_seg_png_256_256\\train_lab"
test_dataset = "E:\\Road_Crack_Dataset\\Crack500_Dataset\\C5_seg_png_256_256\\test_img"
test_label = "E:\\Road_Crack_Dataset\\Crack500_Dataset\\C5_seg_png_256_256\\test_lab"


train_img_plane_folder = "E:\\Road_Crack_Dataset\\Crack500_Dataset\\C5_seg_png_256_256\\train_plane8"
train_img_gray_folder = "E:\\Road_Crack_Dataset\\Crack500_Dataset\\C5_seg_png_256_256\\train_gray"
test_img_plane_folder = "E:\\Road_Crack_Dataset\\Crack500_Dataset\\C5_seg_png_256_256\\test_plane8"
test_img_gray_folder = "E:\\Road_Crack_Dataset\\Crack500_Dataset\\C5_seg_png_256_256\\test_gray"

SEG_NUM = 8
IS_SEGMENT_MOD = False


def process_and_save_planes(image_paths, output_plane_folder, seg_num=SEG_NUM, is_segment_mod=IS_SEGMENT_MOD):

    print(f"Start processing and saving independent binary plane images to: {output_plane_folder}")
    os.makedirs(output_plane_folder, exist_ok=True) # 确保输出目录存在

    count = 0
    total = len(image_paths)
    start_time_batch = tm.time()

    for image_path in image_paths:
        try:
            img_gray = cr5_spt.get_gray_img_from_file(image_path)
            if img_gray is None:
                print(f"  Warning: Unable to load image {image_path}, skipping.")
                continue

            _, threshold_int_list, _, _, _, _ = \
                cr5_spt.img_integer_segmentation_equal_range_thresholds_light(
                    img_gray, seg_num, is_segment_mod=is_segment_mod
                )

            threshold_float_list = threshold_int_list / 255.0
            plane_list_np = cr5_spt.img_segmentation_threshold_list_light(
                img_gray, threshold_float_list
            )

            base_filename = os.path.basename(image_path)
            name, _ = os.path.splitext(base_filename)

            for plane_idx, plane_np in enumerate(plane_list_np):
                 output_filename = os.path.join(output_plane_folder, f"{name}-plane{plane_idx:02d}.png")

                 plane_to_save = (plane_np * 255).astype(np.uint8)

                 success = cv2.imwrite(output_filename, plane_to_save)
                 if not success:
                     print(f"  Error: Unable to save floor plan {output_filename}")

            count += 1
            if count % 10 == 0:
                 elapsed_time = tm.time() - start_time_batch
                 print(f"  Processed {count}/{total} images... (Took: {elapsed_time:.2f} seconds)")
                 start_time_batch = tm.time()

        except Exception as e:
            print(f"  An error occurred while processing image {image_path}: {e}")

    print(f"Processing completed. Floorplans of {count} images saved to {output_plane_folder}")

if __name__ == "__main__":

    try:
        print("Load training set image path...")
        train_image_paths = cr5_spt.load_images(train_dataset)
        print(f"Find {len(train_image_paths)} training images.")

        print("Load the test set image path...")
        test_image_paths = cr5_spt.load_images(test_dataset)
        print(f"Find {len(test_image_paths)} test images.")
    except Exception as e:
         print(f"Error loading image path: {e}")
         exit()


    process_and_save_planes(train_image_paths, train_img_plane_folder, seg_num=SEG_NUM)


    process_and_save_planes(test_image_paths, test_img_plane_folder, seg_num=SEG_NUM)

    print("All preprocessing is done!")
