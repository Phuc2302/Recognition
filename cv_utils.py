import cv2
import os
import numpy as np

DEBUG = False


def set_debug(debug):
    global DEBUG
    DEBUG = debug


def wider_than(contour, min_width):
    # Trả về True nếu đường bao rộng hơn chiều rộng tối thiểu
    x, y, w, h = cv2.boundingRect(contour)
    return w > min_width


def move_bounding_rect(rect, delta_x, delta_y):
    """ Move bounding rect x,y by some delta_x, delta_y .  """
    x, y, w, h = rect
    return (x+delta_x, y+delta_y, w, h)


def concatenate_bounding_rects(bounding_rects):
    # Kết hợp thành một hộp lớn có giới hạn
    temp_arr = []
    for x, y, w, h in bounding_rects:
        temp_arr.append((x, y))
        temp_arr.append((x+w, y+h))

    return cv2.boundingRect(np.asarray(temp_arr))


def get_bounding_rect_content(img, bounding_rect):
    x, y, w, h = bounding_rect
    return img[y:y+h, x:x+w]


def get_contour_area_from_img(img, contour):
    return get_bounding_rect_content(img, cv2.boundingRect(contour))


def get_rotated_image_from_contour(img, contour):

    rotated_rect = cv2.minAreaRect(contour)

    # Lấy tâm x, y và chiều rộng và chiều cao.
    x_center = int(rotated_rect[0][0])
    y_center = int(rotated_rect[0][1])
    width = int(rotated_rect[1][0])
    height = int(rotated_rect[1][1])
    angle_degrees = rotated_rect[2]

    if(width > height):
        temp_height = height
        height = width
        width = temp_height
        angle_degrees =   angle_degrees - 90

    # Gán lại trực tràng đã xoay với các giá trị cập nhật
    rotated_rect = ((x_center, y_center), (width, height), angle_degrees)
    # Tìm tọa độ 4 (x, y) cho hình chữ nhật xoay, theo thứ tự: bl, tl, tr, br
    rect_box_points = cv2.boxPoints(rotated_rect)

    img_debug_contour = img.copy()
    cv2.drawContours(img_debug_contour, [contour], 0, (0, 0, 255), 3)
    show_window('biggest_contour', img_debug_contour)

    img_debug = img.copy()
    cv2.drawContours(img_debug, [np.int0(rect_box_points)], 0, (0, 0, 255), 3)
    show_window('min_area_rect_original_image', img_debug)

    # Chuẩn bị cho chuyển đổi xoay
    src_pts = rect_box_points.astype("float32")
    dst_pts = np.array([
        [1, height-1],  # Bottom Left
        [0, 0],  # Top Left
        [width-1, 0],  # Top Right
    ], dtype="float32")

    # Chuyển đổi xoay vòng liên kết
    ROTATION_MAT = cv2.getAffineTransform(src_pts[:3], dst_pts)
    return cv2.warpAffine(img, ROTATION_MAT, (width, height))


def get_com_shift(img):
    """ Lấy tam giác x và y từ tâm dựa trên khối lượng. """
    M = cv2.moments(img)
    height, width = img.shape
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    shift_x = np.round(width/2-cX).astype(int)
    shift_y = np.round(height/2-cY).astype(int)
    return shift_x, shift_y


def shift_by(img, delta_x, delta_y):
    """ Trả về trung tâm hình ảnh đã dịch theo delta_x và delta_y. """
    rows, cols = img.shape
    M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    return cv2.warpAffine(img, M, (cols, rows))


def draw_bounding_rects(img, bounding_rects):
    """ Vẽ các hình chữ nhật và chỉ số của chúng dựa trên hình ảnh. """
    for index, cell in enumerate(bounding_rects):
        x, y, w, h = cell
        cv2.putText(img, str(index), (x, y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, .6, (100, 200, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, cell, (0, 255, 0), 1)


def get_external_contours(img_1_channel):
    """ Sử dụng hàm findContours của OpenCV """
    # Lấy các đường viền bên ngoài
    contours, _ = cv2.findContours(img_1_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_contours(img_1_channel):

    contours, hierarchy = cv2.findContours(
        img_1_channel, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def __compareBoundingSize(contour):
    contour = cv2.convexHull(contour)
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return w*h


def get_biggest_contour(contours):
    sorted_contours = sorted(contours, key=__compareBoundingSize, reverse=True)
    biggest_contour = sorted_contours[0]
    return biggest_contour


def get_biggest_intensity_contour(contours):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    biggest_contour = sorted_contours[0]
    return biggest_contour


def show_window(name, image, debug=False):
    cv2.imwrite("./assets/output/" + name + ".png", image)
    if debug or DEBUG:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, (700, 700))
        cv2.imshow(name, image)
        cv2.waitKey(0)
