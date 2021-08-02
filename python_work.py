import tensorflow as tf
import argparse
import pytesseract
import numpy as np
import time

import cv2
import cv_utils
import sheet

start = time.time()

print('Reading tensorflow model...')

predict_model = tf.keras.models.load_model('./models/model_tensorflow')
predict_model.summary()

start_proc = time.time()
# img_path = 'D:/Study/AI/Python/NhanDang/Main/digits.jpg'
img_path = './assets/test/sample_left.jpg'

end_read = time.time()
print("Time to read the image: ")
print("---- %s seconds ----" % (end_read - start))

num_rows = 20

parser = argparse.ArgumentParser()

parser.add_argument("--num_rows", help="set num rows of the grid")
parser.add_argument("--img_path", help="specify path to input image")
parser.add_argument("--debug", help="specify debug to stop at show_window")

args = parser.parse_args()

if args.num_rows:
    num_rows = int(args.num_rows)

if args.debug:
    cv_utils.set_debug(bool(args.debug))

if args.img_path:
    img_path = args.img_path

# Step 1
print("Reading image from path", img_path)
input_img = cv2.imread(img_path)#, cv2.IMREAD_GRAYSCALE)

# Step 2
img_sheet, img_binary_sheet, img_binary_only_numbers, cells_bounding_rects = sheet.generate_sheet(input_img, num_rows_in_grid=num_rows)

img_cells = img_sheet.copy()
cv_utils.draw_bounding_rects(img_cells, cells_bounding_rects)
cv_utils.show_window('img_cells', img_cells)

# Step 3
digit_contours = cv_utils.get_external_contours(img_binary_only_numbers)

start_re = time.time()
for i, cnt in enumerate(digit_contours):

    # Step 4
    digit_bounding_rect = cv2.boundingRect(cnt)
    x, y, w, h = digit_bounding_rect

    # Xác định xem các ô nào thuộc về ô bao quanh này
    cell = sheet.validate_and_find_cell(cells_bounding_rects, digit_bounding_rect)
    if cell is None:
        continue

    # Nhị phân đen/trắng của ROI
    roi = img_binary_sheet[y:y + h, x:x + w]

    roi_fit_20x20 = 20 / max(roi.shape[0], roi.shape[1])

    # Thay đổi kích thước duy trì định dạng nhị phân với INTER_NEAREST
    roi = cv2.resize(roi, None, fx=roi_fit_20x20, fy=roi_fit_20x20, interpolation=cv2.INTER_NEAREST)

    roi_background = np.zeros((28, 28), dtype=roi.dtype)
    # Đặt các chữ số vào nền của ROI từ trên xuống và từ trái
    roi_background[4:4 + roi.shape[0], 4:4 + roi.shape[1]] = roi

    # Lưu bản gốc của ROI
    cv2.imwrite("./assets/roi/original/roi_" +
                str(i) + ".png", roi_background)

    # Nhân bản dịch dựa trên trọng tâm
    delta_x, delta_y = cv_utils.get_com_shift(roi_background)

    # Thay đổi ROI theo định dạng
    roi_background = cv_utils.shift_by(roi_background, delta_x, delta_y)
    cv2.imwrite("./assets/roi/shifted/roi_" +
                str(i) + ".png", roi_background)

    # Tiền xử lý để dự đoán
    roi_background = roi_background - 127.5
    roi_background /= 127.5

    # Step 5
    # Xác xuất mất các ký tự từ quá trình phân loại
    prediction = predict_model(np.reshape(roi_background, (1, 28, 28, 1)))

    predicted_digit = np.argmax(prediction)


    # Đánh dấu chúng trên hình ảnh
    cv2.rectangle(img_sheet, (x, y), (x + w, y + h), (100, 10, 100), 1)

    cv2.putText(img_sheet, str(predicted_digit), (x + int(w / 2) + 40, y + int(h / 30) + 30), cv2.FONT_HERSHEY_SIMPLEX, .7,
                (0, 0, 255), 2, cv2.LINE_AA)

end_re = time.time()
print("Time to recognition the image: ")
print("---- %s seconds ----" % (end_re - start_re))
    # import csv
    #
    # # import os
    # #
    # # if os.path.exists('./assets/output_data/data.csv'):
    # #     os.remove('./assets/output_data/data.csv')
    # # else:
    # #     print("The file does not exists")
    #
    # arr = np.array([predicted_digit])
    #
    # outfile = open('./assets/output_data/data.csv', 'a')
    # out = csv.writer(outfile)
    # out.writerows(map(lambda x: [x], reversed(arr)))
    # outfile.close()

cv_utils.show_window('img_sheet', img_sheet, debug=True)
cv_utils.show_window('img_binary_sheet', img_binary_sheet)
cv_utils.show_window('img_binary_only_numbers', img_binary_only_numbers)

end = time.time()
print("Time to finish: ")
print("---- %s seconds ----" % (end - start))