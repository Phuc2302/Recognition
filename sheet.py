from functools import cmp_to_key
import cv2
import numpy as np
import cv_utils
import time

MIN_CELL_DIGIT_HEIGHT_RATIO = 2.5


def __should_merge_lines(line_a, line_b, rho_distance, theta_distance):
    rho_a, theta_a = line_a[0].copy()
    rho_b, theta_b = line_b[0].copy()
    if (rho_b == rho_a and theta_b == theta_b):
        return False

    # Sử dụng mức độ để định dạng người dùng trực quan hơn
    theta_b = int(180 * theta_b / np.pi)
    theta_a = int(180 * theta_a / np.pi)

    # In Q3 or Q4, phương pháp merge_lines
    if rho_b < 0:
        theta_b = theta_b - 180

    # In Q3 or Q4, phương pháp merge_lines
    if rho_a < 0:
        theta_a = theta_a - 180

    rho_a = np.abs(rho_a)
    rho_b = np.abs(rho_b)

    diff_theta = np.abs(theta_a - theta_b)
    rho_diff = np.abs(rho_a - rho_b)

    if (rho_diff < rho_distance and diff_theta < theta_distance):
        return True
    return False

def resize_to_right_ratio(img, interpolation=cv2.INTER_LINEAR, width=1550):
    ratio_width = width / img.shape[1]
    # Resize
    return cv2.resize(img, None, fx=ratio_width, fy=ratio_width, interpolation=interpolation)


def merge_lines(line_a, line_b):
    rho_b, theta_b = line_b[0]
    rho_a, theta_a = line_a[0]

    #
    # Chuyển theta thành âm (trong Q3 sang Q4)
    if rho_b < 0:
        rho_b = np.abs(rho_b)
        theta_b = theta_b - np.pi

    #
    # Chuyển theta thành âm (trong Q3 sang Q4)
    if rho_a < 0:
        rho_a = np.abs(rho_a)
        theta_a = theta_a - np.pi

    average_theta = (theta_a + theta_b) / 2
    average_rho = (rho_a + rho_b) / 2
    # Trong Q3 hoặc Q4, sau khi tính trung bình cả hai dòng, định dạng sang định dạng OpenCV HoughLines
    if average_theta < 0:
        # rho là giá trị âm
        average_rho = -average_rho
        # Định dạng lại thành giá trị dương cho opencv HoughLines 0 thành PI
        average_theta = np.abs(average_theta)

    return [[average_rho, average_theta]]

def get_merged_line(lines, line_a, rho_distance, degree_distance):

    for i, line_b in enumerate(lines):
        if line_b is False:
            continue
        if __should_merge_lines(line_a, line_b, rho_distance, degree_distance):
            # Update line A on every iteration
            line_a = merge_lines(line_a, line_b)
            # Don't use B again
            lines[i] = False

    return line_a


# def get_adaptive_binary_image(img):
#     # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #img = (cv2.GaussianBlurimg, (5, 5), 0)
#
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
#     return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7,10)

def get_adaptive_binary_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return cv2.adaptiveThreshold(gray, 255,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 4)

def merge_nearby_lines(lines, rho_distance=15, degree_distance=20):

    lines = lines if lines is not None else []
    estimated_lines = []
    for line in lines:
        if line is False:
            continue

        estimated_line = get_merged_line(lines, line, rho_distance, degree_distance)
        estimated_lines.append(estimated_line)

    return estimated_lines

def draw_lines(lines, img):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
            pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
            cv2.line(img, pt1, pt2,
                     (255, 255, 255), 2)

def sort_by_upper_left_pos(rect_a, rect_b):
    x_a, y_a, _, _ = rect_a
    x_b, y_b, w_b, _ = rect_b

    # Xử lý phân tán
    x_b_offset_positive = x_b + w_b / 3
    x_b_offset_negative = x_b - w_b / 3
    is_same_column = x_a < x_b_offset_positive and x_a > x_b_offset_negative

    if is_same_column:
        return y_a - y_b
    return (x_a - x_b_offset_positive)

def get_rotated_sheet(img, img_binary):
    # Tìm đường bao bên ngoài lớn nhất để định vị trang tính. Sử dụng RETR_EXTERNAL và loại bỏ các đường bao lồng nhau.
    contours = cv_utils.get_external_contours(img_binary)
    biggest_contour = cv_utils.get_biggest_intensity_contour(contours)

    img_raw_sheet = cv_utils.get_rotated_image_from_contour(img, biggest_contour)

    # img_raw_sheet = resize_to_right_ratio(img_raw_sheet)
    img_binary_sheet_rotated = get_adaptive_binary_image(img_raw_sheet)

    return img_raw_sheet, img_binary_sheet_rotated


def get_grid(img_binary_sheet):
    height, width = img_binary_sheet.shape

    img_binary_sheet_morphed = img_binary_sheet.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_binary_sheet_morphed = cv2.morphologyEx(img_binary_sheet_morphed, cv2.MORPH_DILATE, kernel)

    cv_utils.show_window('morph_dilate_binary_img', img_binary_sheet_morphed)

    sheet_binary_grid_horizontal = img_binary_sheet_morphed.copy()
    sheet_binary_grid_vertical = img_binary_sheet_morphed.copy()

    # Chúng tôi sử dụng độ dài tương đối cho dòng cấu trúc để linh hoạt cho nhiều kích thước của trang tính.
    structuring_line_size = int(width / 5.0)

    # Xóa tất cả nội dung theo chiều dọc trong hình ảnh
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (structuring_line_size, 1))
    sheet_binary_grid_horizontal = cv2.morphologyEx(sheet_binary_grid_horizontal, cv2.MORPH_OPEN, element)

    # Xóa tất cả nội dung nằm ngang trong hình ảnh, Morph OPEN: Giữ mọi thứ phù hợp với yếu tố cấu trúc,
    # tức là các đường thẳng đứng
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, structuring_line_size))

    sheet_binary_grid_vertical = cv2.morphologyEx(sheet_binary_grid_vertical, cv2.MORPH_OPEN, element)

    # Nối các đường dọc / ngang thành lưới
    img_binary_sheet_morphed = cv2.add(sheet_binary_grid_vertical, sheet_binary_grid_horizontal)

    cv_utils.show_window("morph_keep_only_horizontal_lines",
                         sheet_binary_grid_horizontal)
    cv_utils.show_window("morph_keep_only_vertical_lines", sheet_binary_grid_vertical)
    cv_utils.show_window("concatenate_vertical_horizontal", img_binary_sheet_morphed)

    rho_accumulator = 1
    angle_accumulator = np.pi / 2
    # Bỏ phiếu tối thiểu để xác định một dòng
    threshold_accumulator_votes = int(width / 2)

    # Tìm các dòng trong hình ảnh theo Thuật toán Hough
    grid_lines = cv2.HoughLines(img_binary_sheet_morphed, rho_accumulator,
                                angle_accumulator, threshold_accumulator_votes)

    img_binary_grid = np.zeros(
        img_binary_sheet_morphed.shape, dtype=img_binary_sheet_morphed.dtype)

    # Có thể có nhiều dòng cho cùng một đường lưới, nên hợp nhất các dòng lân cận
    grid_lines = merge_nearby_lines(grid_lines)
    draw_lines(grid_lines, img_binary_grid)

    # Vì tất cả các trang tính không có viền ngoài. Nên vẽ một hình chữ nhật xung quanh
    outer_border = np.array([
        [1, height - 1],  # Bottom Left
        [1, 1],  # Top Left
        [width - 1, 1],  # Top Right
        [width - 1, height - 1]  # Bottom Right
    ])
    cv2.drawContours(img_binary_grid, [outer_border], 0, (255, 255, 255), 3)

    # Xóa lưới khỏi ảnh nhị phân và chỉ giữ lại các chữ số.
    img_binary_sheet_only_digits = cv2.bitwise_and(img_binary_sheet, 255 - img_binary_sheet_morphed)

    cv_utils.show_window("grid_binary_lines", img_binary_grid)

    return   img_binary_grid, img_binary_sheet_only_digits


def __filter_by_dim(val, target_width, target_height):
    # Loại bỏ các ô bên ngoài chiều rộng / chiều cao mục tiêu
    offset_width = target_width * 0.6
    offset_height = target_height * 0.6
    _, _, w, h = val
    return target_width - offset_width < w < target_width + offset_width and target_height - offset_height < h < target_height + offset_height


def __get_most_common_area(bounding_rects, cell_resolution):
    cell_areas = [int(w*h/cell_resolution) for _, _, w, h in bounding_rects]

    # 1-Dimensional groupsgenerate_ysheet
    counts = np.bincount(cell_areas)
    return bounding_rects[np.argmax(counts)]


def validate_and_find_cell(cells, bounding_rect):
    # Tìm ô tương ứng cho bounding_rect.
    roi_x, roi_y, roi_w, roi_h = bounding_rect
    roi_center_x = roi_x + int(roi_w/2)
    roi_center_y = roi_y + int(roi_h/2)
    _, _, cell_width, cell_height = cells[0]
    # Loại bỏ các đường viền nhiễu
    if(not 2 < roi_w < cell_width or not (cell_height/MIN_CELL_DIGIT_HEIGHT_RATIO) < roi_h < cell_height):
        return None

    found_cell = None
    for rect in cells:
        x, y, w, h = rect
        # Verify which cell roi_center belongs to
        if(x < roi_center_x < x + w and y < roi_center_y < y + h):
            found_cell = rect
            break

    return found_cell

def get_cells_bounding_rects(img_binary_grid, num_rows_in_grid=36, max_num_cols=20):

    binary_grid_contours, _ = cv2.findContours(img_binary_grid, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

    sheet_width = img_binary_grid.shape[1]

    cell_min_width = (sheet_width/max_num_cols)

    # Làm sạch lưới từ các đường viền nhỏ
    cells_bounding_rects = [cv2.boundingRect(cnt) for cnt in binary_grid_contours if cv_utils.wider_than(cnt, cell_min_width)]

    # Xác định độ phân giải cho "bên trong" khu vực ô
    cell_resolution = (sheet_width/50) ** 2

    _, _, target_width, target_height = __get_most_common_area(cells_bounding_rects, cell_resolution)

    if len(cells_bounding_rects) < num_rows_in_grid:
        print("ERROR: Not enough grid cells found.")

    cells_bounding_rects = list(filter(lambda x: __filter_by_dim(x, target_width, target_height), cells_bounding_rects))

    num_cells = len(cells_bounding_rects)
    correct_num_cells_in_grid = (num_cells >= num_rows_in_grid and num_cells % num_rows_in_grid == 0)

    if not correct_num_cells_in_grid:
        print("ERROR: not correct number for cells found in grid, num found:", num_cells)

    grid_bounding_rect = cv_utils.concatenate_bounding_rects(cells_bounding_rects)

    shift_x, shift_y, _, _ = grid_bounding_rect

    cells_bounding_rects = list(map(lambda x: cv_utils.move_bounding_rect(x, -shift_x, -shift_y), cells_bounding_rects))
    cells_bounding_rects = sorted(cells_bounding_rects, key=cmp_to_key(sort_by_upper_left_pos))
    return cells_bounding_rects, grid_bounding_rect

def generate_sheet(img, num_rows_in_grid=37, max_num_cols=20):
    start = time.time()

    img = resize_to_right_ratio(img)

    end_resize = time.time()
    print("=================================================")
    print("Time to resize the image: ")
    print("---- %s seconds ----" % (end_resize - start))
    # Step 1
    img_adaptive_binary = get_adaptive_binary_image(img)

    cv_utils.show_window('img_adaptive_binary', img_adaptive_binary)

    end_binary = time.time()
    print("=================================================")
    print("Time to binary the image: ")
    print("---- %s seconds ----" % (end_binary - start))

    # Step2 and 3, Tìm đường viền lớn nhất và xoay nó
    img_sheet, img_binary_sheet = get_rotated_sheet(img, img_adaptive_binary)

    cv_utils.show_window('img_sheet_original', img_sheet)

    end_fContour = time.time()
    print("=================================================")
    print("Time to biggest contour the image: ")
    print("---- %s seconds ----" % (end_fContour - start))

    # Step 4, Nhận một lưới mới với các đường dọc / ngang
    img_binary_grid, img_binary_only_numbers = get_grid(img_binary_sheet)

    end_gird = time.time()
    print("=================================================")
    print("Time to draw gird the image: ")
    print("---- %s seconds ----" % (end_gird - start))

    # Step 5, Nhận mọi ô lưới dưới dạng một lưới giới hạn được sắp xếp để sau này xác định vị trí các số để sửa ô
    cells_bounding_rects, grid_bounding_rect = get_cells_bounding_rects(img_binary_grid, num_rows_in_grid, max_num_cols)

    # Nhận diện tích của lưới từ các phiên bản khác nhau của raw img
    img_binary_only_numbers = cv_utils.get_bounding_rect_content(img_binary_only_numbers, grid_bounding_rect)
    img_binary_sheet = cv_utils.get_bounding_rect_content(img_binary_sheet, grid_bounding_rect)
    img_sheet = cv_utils.get_bounding_rect_content(img_sheet, grid_bounding_rect)

    end = time.time()
    print("=================================================")
    print("Time to locate the image: ")
    print("---- %s seconds ----" % (end - start))

    return img_sheet, img_binary_sheet, img_binary_only_numbers, cells_bounding_rects