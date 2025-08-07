import crack500_support as cr5_spt

train_img_plane_folder = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/train_plane8"
# train_img_gray_folder = "D:/PycharmProjects/test/crack_python2/datasets/train_img_gray"
test_img_plane_folder = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/test_plane8"
# test_img_gray_folder = "D:/PycharmProjects/test/crack_python2/datasets/test_img_gray"
# train_image_gray_paths = cr5_spt.load_images(train_img_gray_folder)
train_image_plane_paths = cr5_spt.load_images(train_img_plane_folder)
# test_image_gray_paths = cr5_spt.load_images(test_img_gray_folder)
test_image_plane_paths = cr5_spt.load_images(test_img_plane_folder)
train_source_tensor = cr5_spt.load_images_as_tensors(train_image_plane_paths)
train_target_tensor = cr5_spt.load_images_as_tensors(train_image_plane_paths)
test_source_tensor = cr5_spt.load_images_as_tensors(test_image_plane_paths)
test_target_tensor = cr5_spt.load_images_as_tensors(test_image_plane_paths)

train_image_names = cr5_spt.get_img_name(train_image_plane_paths)
train_target_names = cr5_spt.get_img_name(train_image_plane_paths)
test_image_names = cr5_spt.get_img_name(test_image_plane_paths)
test_target_names = cr5_spt.get_img_name(test_image_plane_paths)
print(len(train_source_tensor))
print(len(test_source_tensor))
