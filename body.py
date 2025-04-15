import cv2
import numpy as np
import os
def body(image_path):
    # # Load binary image
    # binary_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # # 计算欧氏距离
    # dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 3)
    #
    # # 归一化距离图像
    # cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    #
    # # 将距离图像转换为平滑像素值图像
    # # smoothed = np.exp(-dist ** 2 / 0.1)
    # smoothed = 1 - smoothed
    # # 显示结果
    # cv2.imshow('Smoothed Image', smoothed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # 计算欧式距离变换
    # dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 3)
    #
    # # 将距离值归一化到0-255范围内
    # max_distance = np.max(dist_transform)
    #
    #
    # normalized_dist = dist_transform / max_distance  * 255
    #
    # # smoothed = np.exp(-normalized_dist ** 2 / 0.3)
    # # smoothed = (1 - smoothed) * 255
    # normalized_dist_ = normalized_dist + ( (255 - normalized_dist) / 255 * normalized_dist)
    # # 将距离值转换为像素值
    # pixel_values = normalized_dist_.astype(np.uint8)
    # # pixel_values[pixel_values<=30] = 0
    # # 可视化像素值缓慢变化的图像
    #
    # cv2.imshow('Slowly Changing Pixel Values Image', pixel_values)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print('Error: could not load image')
        exit()

    # Calculate Euclidean distance transform
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    if dist_transform is None:
        print('Error: could not compute distance transform')
        exit()

    # Normalize distance transform to range [0, 255]
    dist_transform_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)

    if dist_transform_norm is None:
        print('Error: could not normalize distance transform')
        exit()

    # Convert normalized distance transform to grayscale image
    dist_transform_norm = dist_transform_norm + ((255 - dist_transform_norm) / 255 * dist_transform_norm)
    skeleton_map = np.uint8(dist_transform_norm)


    return skeleton_map, img - skeleton_map

    # if skeleton_map is None:
    #     print('Error: could not convert distance transform to grayscale')
    #     exit()
    #
    # # Dilate skeleton by a 3x3 kernel
    # kernel = np.ones((3, 3), np.uint8)
    # dilated_skeleton = cv2.dilate(skeleton_map, kernel, iterations=1)
    #
    # if dilated_skeleton is None:
    #     print('Error: could not dilate skeleton')
    #     exit()
    #
    # return dilated_skeleton, img - dilated_skeleton

if __name__=='__main__':

    data_path = '../zzs/'
    save_path = '../zzs/'
    lines = os.listdir(data_path)
    for line in lines:
        image_path = data_path + line
        image_path = '../zzs/1.png'
        gt_body, gt_edge = body(image_path)

        cv2.imwrite('../zzs/res1.jpg', gt_body)
    # Display dilated skeleton
    # cv2.imshow('Dilated Skeleton', dilated_skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# # Load binary image
# img = cv2.imread('./0159.jpg', cv2.IMREAD_GRAYSCALE)
#
# if img is None:
#     print('Error: could not load image')
#     exit()
#
# # Calculate Euclidean distance transform
# dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
#
# if dist_transform is None:
#     print('Error: could not compute distance transform')
#     exit()
#
# # Normalize distance transform to range [0, 255]
# dist_transform_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
#
# if dist_transform_norm is None:
#     print('Error: could not normalize distance transform')
#     exit()
#
# # Convert normalized distance transform to grayscale image
# skeleton_map = np.uint8(dist_transform_norm)
#
# if skeleton_map is None:
#     print('Error: could not convert distance transform to grayscale')
#     exit()
#
# # Display skeleton map
# cv2.imshow('Skeleton Map', skeleton_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()









#
# import cv2
# import numpy as np
#
# # Load binary image
# img = cv2.imread('./0159.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Calculate Euclidean distance transform
# dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
#
# # Normalize distance transform to range [0, 1]
# dist_transform_norm = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
#
# # Convert normalized distance transform to grayscale image
# skeleton_map = np.uint8(255 * dist_transform_norm)
#
# # Display skeleton map
# cv2.imshow('Skeleton Map', skeleton_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()