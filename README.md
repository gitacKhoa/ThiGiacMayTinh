# Tolerance Vision

Pipeline Python/OpenCV cho ảnh vật trên mặt phẳng chuẩn:

```text
Ảnh gốc + calibration
  → undistort (alpha=0, crop một lần, không resize)
  → chọn ROI vật trên ảnh undistort
  → segmentation → mask toàn ảnh
  → Canny trên mức xám + giới hạn quanh biên mask
  → tinh chỉnh cực đại gradient theo pháp tuyến → điểm sub-pixel (x, y)
  → homography từ bàn cờ trong CÙNG ảnh → tọa độ mm (tùy chọn)
```

Đã triển khai tới đầu ra tọa độ biên. [Đo kích thước tự động](src/measurement/measure.py),
[ước lượng độ không đảm bảo](src/measurement/uncertainty.py), [QC](src/qc/tolerance_check.py)
và [GUI](src/gui/app.py) hiện vẫn là file trống.
Xem [báo cáo rà soát code](docs/code-review-flow.md) để biết các lỗi đã sửa và phần còn thiếu.

## Cài đặt

```powershell
python -m pip install -r requirements.txt
```

Hai module mới chỉ sử dụng NumPy và OpenCV, không thêm thư viện.
Môi trường kiểm tra: Python 3.11.9, OpenCV 4.11.0, NumPy 2.2.4.
Chạy các lệnh dưới đây từ thư mục gốc dự án.

## Chạy ảnh mẫu đã có

Ví dụ chọn vật màu xanh bên trái bàn cờ trong [ảnh thử](data/calibration_test/homography_test.jpg):

```powershell
python -m src.pipeline `
  --image data/calibration_test/homography_test.jpg `
  --calib outputs/calibration_results/calibration_data.pkl `
  --calibration-size 1280 720 `
  --roi 100 80 380 480 `
  --method hsv --hsv-lower 85 80 35 --hsv-upper 140 255 255 `
  --fill-holes --canny-low 10 --canny-high 30 `
  --square-mm 29 --cols 8 --rows 5 `
  --output outputs/edge_pipeline_demo
```

Thông số 29 mm theo cấu hình bàn cờ hiện có; cần đo cạnh ô in thực tế để xác nhận thang đo.
Đã kiểm tra 15 ảnh calibration đều có kích thước 1280×720. File calibration cũ chưa có
`image_size`, vì vậy phải truyền `--calibration-size`. File calibration tạo mới sẽ lưu trường này.
Ảnh có kích thước khác calibration sẽ bị từ chối; không tự co giãn ma trận camera.

Kết quả mẫu: 1.226 điểm, RMSE khớp bàn cờ khoảng 0,1315 mm.
Đây là sai số khớp các góc bàn cờ, **không phải sai số đo vật hoặc độ không đảm bảo**.
Cạnh trên vật có đoạn thiếu và cạnh trái còn nhiễu do nền/họa tiết/độ tương phản thấp;
kết quả là minh họa nối flow, chưa xác nhận độ chính xác đo sản phẩm.

## Chạy ảnh vật mới

Vật tối trên nền sáng, chọn ROI bằng chuột:

```powershell
python -m src.pipeline `
  --image captured.jpg `
  --calib outputs/calibration_results/calibration_data.pkl `
  --calibration-size 1280 720 `
  --select-roi --method otsu --foreground dark `
  --output outputs/my_object
```

Chọn hình chữ nhật chứa toàn bộ vật và một khoảng nền quanh vật, nhấn Enter/Space.
ROI `(x, y, width, height)` luôn thuộc **ảnh đã undistort/crop hiển thị lúc chọn**.
Tránh đưa bàn cờ hoặc vật khác vào ROI. Đối tượng chạm mép ROI mặc định bị loại.

- Vật sáng trên nền tối: `--foreground bright`.
- Ánh sáng không đều: thử `--method adaptive --adaptive-block-size 31 --adaptive-c 5`;
  cần kiểm tra mask vì adaptive threshold có thể tạo vùng dạng vành ở vật lớn.
- Tách màu: `--method hsv --hsv-lower H S V --hsv-upper H S V`.
  Hue dùng thang OpenCV 0–179; S/V 0–255. Hue thấp > hue cao sẽ chọn qua mốc 0.
- `--fill-holes` dùng khi chữ/họa tiết tạo lỗ giả trên một vật đặc.
  Không bật nếu cần giữ lỗ thật của chi tiết.
- `--all-components` giữ mọi thành phần đủ diện tích trong ROI, mặc định giữ lớn nhất.
- `--blur-size 1 --morphology-size 1` tắt làm mượt/cleanup mask.
- `--boundary-radius 3` chọn dải ứng viên quanh biên mask. Tăng quá lớn có thể nhận họa tiết/bóng.
- `--canny-low`, `--canny-high`, `--sigma`, `--min-gradient`, `--max-offset` điều khiển bước biên.
- Thêm `--square-mm 29 --cols 8 --rows 5` để xuất mm. Nếu không tìm thấy bàn cờ,
  chương trình báo lỗi thay vì xuất mm sai. Không truyền `--square-mm` thì chỉ xuất pixel.
- Ảnh đã undistort: dùng `--already-undistorted` thay `--calib` để tránh sửa méo hai lần.

## Các file đầu ra

Mỗi lần chạy ghi các file sau trong thư mục `--output` (chạy lại cùng thư mục sẽ thay kết quả):

| File | Nội dung |
| --- | --- |
| `undistorted.png` | Ảnh dùng chung cho toàn bộ phép tính tọa độ |
| `mask.png` | Mask nhị phân uint8 0/255, cùng kích thước ảnh |
| `edges.png` | Pixel ứng viên đã vượt kiểm tra tinh chỉnh |
| `overlay.png` | Biên sub-pixel vẽ lên ảnh để kiểm tra |
| `edges.csv` | x/y pixel, độ dịch theo pháp tuyến, độ mạnh gradient, x/y mm nếu có |
| `edges.npz` | Dữ liệu float64: points_px, pixel_points, normals, offsets, strengths, points_mm, homography |
| `metadata.json` | Tham số chạy, ROI/crop, hệ tọa độ, số điểm và RMSE bàn cờ |

Tập điểm sub-pixel **không được sắp theo contour**. Không đưa trực tiếp vào `contourArea`
hoặc `arcLength`. Bước đo tiếp theo cần chọn đoạn biên phù hợp rồi fit đường thẳng/đường tròn
hoặc xây dựng contour có thứ tự theo loại kích thước cần đo.

## Dùng các module trong Python

```python
from src.calibration.undistort import load_calibration_data, undistort_image
from src.segmentation.segment import segment_image
from src.edge_detection.subpixel_edge import detect_subpixel_edges
from src.measurement.homography import compute_homography, points_to_mm

# raw_image: ảnh uint8 BGR đọc từ camera/file; ví dụ dùng calibration cũ 1280x720.
calibration = load_calibration_data("outputs/calibration_results/calibration_data.pkl")
image = undistort_image(
    raw_image, calibration["mtx"], calibration["dist"],
    calibration_size=calibration.get("image_size", (1280, 720)),
)
segment = segment_image(image, roi=(100, 80, 380, 480), method="hsv",
                        hsv_lower=(85, 80, 35), hsv_upper=(140, 255, 255), fill_holes=True)
edges = detect_subpixel_edges(image, segment.mask, roi=segment.roi,
                              canny_low=10, canny_high=30)
H, corners = compute_homography(image, square_mm=29, pattern_size=(8, 5))
if H is not None:
    points_mm = points_to_mm(edges.points, H)
```

## Thuật toán sub-pixel

1. Làm mượt mức xám bằng Gaussian; Sobel tính gradient và hướng pháp tuyến.
2. Canny cung cấp ứng viên nguyên pixel. Mask chỉ giới hạn ứng viên quanh biên vật.
3. Nội suy song tuyến tính độ lớn gradient ở `p-n`, `p`, `p+n`.
4. Fit parabol ba mẫu và lấy đỉnh:
   `delta = 0.5*(g_minus-g_plus)/(g_minus-2*g_center+g_plus)`;
   `p_subpixel = p + delta*n`.
5. Loại profile không có cực đại, gradient yếu, thiếu vùng nội suy hoặc độ dịch vượt ngưỡng.
   Không gắn nhãn sub-pixel cho điểm fallback nguyên pixel.

Hàm trả `(N, 2)` kể cả khi không có điểm (`N=0`). Mask không được dùng làm ảnh gradient.
Tọa độ thập phân là kết quả ước lượng vị trí, không tự bảo đảm độ chính xác 0,1 pixel/mm.

## Kiểm tra

```powershell
python -m unittest discover -s tests -p test_vision_pipeline.py -v
```

Kiểm tra gồm polarity/ROI, HSV qua mốc hue, nhiễu/lỗ/vật chạm biên, biên nghiêng và biên cong
có vị trí biết trước, mask không tạo biên giả, calibration NPZ/PKL, kích thước/crop,
homography, flow bàn cờ tổng hợp + vật và xuất file, hồi quy lưu ảnh camera sạch.
Webcam thật và độ chính xác kích thước sản phẩm chưa được kiểm chứng bằng bộ test này.

## Tài liệu API tham khảo

- [OpenCV thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [OpenCV Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [OpenCV calibration và homography](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
