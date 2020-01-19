import os 
import shutil

SOURCE_DIR = 'D:/Datasets/totoro_blackplate2'
DESTINATION_DIR = 'output/totoro_blackplate2_seperate_cam'
TOTAL_CAMERA = 10
REQUIRE_ROTATION = True

if REQUIRE_ROTATION:
    import cv2
    ROTATE_DIRECTION = cv2.ROTATE_90_CLOCKWISE


if not os.path.exists(DESTINATION_DIR):
    os.mkdir(DESTINATION_DIR)

files = os.listdir(SOURCE_DIR)
for camera_number in range(TOTAL_CAMERA):
    camera_files = [f for f in files if f[:3] == "{:03d}".format(camera_number) ]
    camera_dir = os.path.join(DESTINATION_DIR,'cam{:03d}'.format(camera_number))
    if not os.path.exists(camera_dir):
        os.mkdir(camera_dir)    
    for image in camera_files:
        image_ext = image.split('.')[-1]
        image_id = int(image[3:6])
        source_image = os.path.join(SOURCE_DIR,image)
        destination_image = os.path.join(camera_dir,"cam{:03d}_{:05d}.{}".format(camera_number,image_id,image_ext))
        if REQUIRE_ROTATION:
            data = cv2.imread(source_image)
            data = cv2.rotate(data,ROTATE_DIRECTION)
            cv2.imwrite(destination_image,data)
        else:
            shutil.copy2(source_image,destination_image)