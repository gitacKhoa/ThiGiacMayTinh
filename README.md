# Tolerance Vision

Pipeline Python/OpenCV cho ảnh vật trên mặt phẳng chuẩn:

```text
Ảnh gốc + calibration
  → undistort (alpha=0, crop một lần, không resize)
  → chọn ROI vật trên ảnh undistort
  → segmentation → mask toàn ảnh
  → Canny trên mức xám + giới hạn quanh biên mask
  → tinh chỉnh cực đại gradient theo pháp tuyến → điểm sub-pixel (x, y)
  → homography tính mới từ bàn cờ hoặc nạp từ lần trước → tọa độ mm
  → đo hình chữ nhật / đường tròn / kích thước bao ngoài
  → ước lượng độ không đảm bảo → so dung sai → xuất báo cáo + ảnh chú thích
```

Đã triển khai [đo kích thước](src/measurement/measure.py),
[độ không đảm bảo](src/measurement/uncertainty.py), [QC](src/qc/tolerance_check.py),
[xuất báo cáo](src/reporting.py) và [đối chiếu mẫu chuẩn](src/validation.py).
[API xử lý](src/inspection.py) độc lập cửa sổ, camera và ghi file để dùng lại khi xây GUI.
Giao diện chưa triển khai. Xem [hướng dẫn API và cấu hình](docs/inspection-api.md).
[Báo cáo rà soát cũ](docs/code-review-flow.md) là tài liệu lịch sử, không mô tả đầy đủ trạng thái hiện tại.

## Cài đặt

```powershell
python -m pip install -r requirements.txt
```

Các module xử lý chỉ sử dụng NumPy, OpenCV và thư viện chuẩn; không thêm dependency.
Môi trường kiểm tra: Python 3.11.9, OpenCV 4.11.0, NumPy 2.2.4.
Chạy các lệnh dưới đây từ thư mục gốc dự án.

## Chạy ngay ảnh chìa khóa hiện có

```powershell
python -m src.pipeline `
  --image object_undistorted.png --already-undistorted `
  --homography-mode reuse --homography-file outputs/homography.json `
  --roi 492 295 256 153 --measurement bounding-box `
  --output outputs/key_measurement `
  --save-config outputs/key_measurement/inspection_config.json
```

ROI này chỉ dành cho ảnh hiện tại; ảnh khác dùng `--select-roi`.
`bounding-box` đo hình chữ nhật bao nhỏ nhất trong **hệ mm**, không ép chìa khóa thành vật hình chữ nhật.
Kết quả thử hiện tại khoảng 59,121 × 24,717 mm; chưa đối chiếu kích thước thực bằng dụng cụ chuẩn.
Không cung cấp dung sai thì QC là `NOT_CONFIGURED`, không tự kết luận đạt.

## Ví dụ cũ: trích biên vật màu xanh

Ví dụ dưới đây chỉ dùng nếu bạn còn bản ảnh `data/calibration_test/homography_test.jpg`
(ảnh này hiện đã bị xóa khỏi working tree). Đây là chế độ trích biên, không đo kích thước:

```powershell
python -m src.pipeline `
  --image data/calibration_test/homography_test.jpg `
  --calib outputs/calibration_results/calibration_data.pkl `
  --calibration-size 1280 720 `
  --roi 100 80 380 480 `
  --method hsv --hsv-lower 85 80 35 --hsv-upper 140 255 255 `
  --fill-holes --canny-low 10 --canny-high 30 `
  --homography-mode compute --square-mm 29 --cols 8 --rows 5 `
  --homography-file outputs/homography.json `
  --measurement none --output outputs/edge_pipeline_demo
```

Thông số 29 mm theo cấu hình bàn cờ hiện có; cần đo cạnh ô in thực tế để xác nhận thang đo.
Đã kiểm tra 15 ảnh calibration đều có kích thước 1280×720. File calibration cũ chưa có
`image_size`, vì vậy phải truyền `--calibration-size`. File calibration tạo mới sẽ lưu trường này.
Ảnh có kích thước khác calibration sẽ bị từ chối; không tự co giãn ma trận camera.

Kết quả ghi nhận ở phiên bản cũ: 1.226 điểm, RMSE khớp bàn cờ khoảng 0,1315 mm.
Đây là sai số khớp các góc bàn cờ, **không phải sai số đo vật hoặc độ không đảm bảo**.
Cạnh trên vật có đoạn thiếu và cạnh trái còn nhiễu do nền/họa tiết/độ tương phản thấp;
kết quả là minh họa nối flow, chưa xác nhận độ chính xác đo sản phẩm.

## Chạy ảnh vật mới

Ví dụ vật hình chữ nhật tối trên nền sáng, chọn ROI bằng chuột; nạp H đã tính trước đó, ảnh không cần bàn cờ:

```powershell
python -m src.pipeline `
  --image captured.jpg `
  --calib outputs/calibration_results/calibration_data.pkl `
  --calibration-size 1280 720 `
  --homography-mode reuse --homography-file outputs/homography.json `
  --select-roi --method otsu --foreground dark --measurement rectangle `
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
- `--homography-mode compute --square-mm 29 --cols 8 --rows 5`: tính mới và lưu H.
  Không phát hiện bàn cờ thì báo lỗi, không thay H cũ.
- `--homography-mode reuse`: nạp H từ `--homography-file`, không dò bàn cờ.
  Thiếu/hỏng file hoặc kích thước ảnh không khớp thì báo lỗi, không tự chuyển sang pixel.
- Không chỉ định chế độ: có `--square-mm` thì tính mới (tương thích lệnh cũ),
  không có thì nạp lại. Mặc định dùng `outputs/homography.json`.
- Ảnh đã undistort: dùng `--already-undistorted` thay `--calib` để tránh sửa méo hai lần.

## Tính homography một lần, dùng lại nhiều lần

H ánh xạ tọa độ pixel trên **ảnh đã undistort/crop** sang mm trên mặt phẳng bàn cờ.
Gốc tọa độ mm nằm ở góc trong đầu tiên của bàn cờ lúc tính H, vẫn giữ nguyên sau khi bỏ bàn cờ.
Giữ nguyên vị trí/góc camera, focus/zoom, mặt phẳng đo, độ phân giải và cách undistort/crop.
Nếu thay đổi một trong các điều kiện này, phải đặt lại bàn cờ và tính lại H.
Chương trình không tự phát hiện được camera hay mặt phẳng đo đã dịch chuyển.

File JSON lưu H 3×3, đơn vị mm, kích thước ảnh, hệ tọa độ, số góc trong,
kích thước ô, RMSE khớp bàn cờ, thời điểm tính và thông tin undistort/crop nếu có.
Khi cả hai lần đều xử lý ảnh gốc bằng calibration, chương trình kiểm tra thêm độ tương thích
ma trận camera, distortion, crop và kích thước ảnh nguồn. Với ảnh undistort sẵn không có metadata,
chỉ kiểm tra được kích thước ảnh và sẽ nhắc người dùng giữ nguyên cách xử lý.
Module homography không đọc calibration `.pkl`; calibration chỉ phục vụ bước undistort trước đó.

Có thể tạo H ngay lúc chụp webcam (không cần có vật để segmentation):

```powershell
python tests/capture_webcam.py `
  --calib outputs/calibration_results/calibration_data.pkl `
  --calibration-size 1280 720 --width 1280 --height 720 `
  --homography-mode compute --square-mm 29 --cols 8 --rows 5 `
  --homography-file outputs/homography.json `
  --output board_undistorted.png
```

Nhấn Space/C để chụp: nếu không thấy bàn cờ, báo lỗi và cho chụp lại; không lưu ảnh/H mới.
Mỗi lần chụp thành công ở chế độ `compute` cập nhật file H được chỉ định.
Những lần sau bỏ bàn cờ, dùng `--homography-mode reuse`, bỏ `--square-mm/--cols/--rows`
và đổi `--output` sang tên ảnh vật. Có `--calib` mà không chọn chế độ thì mặc định `reuse`.
Không có `--calib` và không chọn chế độ homography thì vẫn chụp ảnh gốc phục vụ calibration.
Script chụp chỉ lưu ảnh và chuẩn bị H; pipeline trích biên, đổi tọa độ và đo kích thước:

```powershell
python -m src.pipeline `
  --image object_undistorted.png --already-undistorted `
  --homography-mode reuse --homography-file outputs/homography.json `
  --select-roi --measurement bounding-box --output outputs/my_object
```

Trong pipeline, H được lưu ngay khi tính thành công, trước bước chọn ROI/tách vật.
Vì vậy vẫn có thể dùng H nếu bước xử lý vật sau đó thất bại. Lần `reuse` không ghi đè file H;
RMSE tham chiếu là của lần tính trước, không phải sai số đo trên ảnh mới.

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
| `metadata.json` | Tham số chạy, ROI/crop, hệ tọa độ, số điểm, chế độ homography và bản sao profile tham chiếu |
| `measurements.json` | Kích thước mm, hình học fit, độ không đảm bảo, giới hạn dung sai, trạng thái QC và cấu hình |
| `measurements.csv` | Mỗi hàng một kích thước, sai lệch, độ không đảm bảo và kết luận |
| `measurement_overlay.png` | Hình học đo vẽ lên ảnh gốc + bảng kích thước riêng, không kéo giãn ảnh |

File `--homography-file` được lưu riêng, không phụ thuộc thư mục `--output`.

Tập điểm sub-pixel **không được sắp theo contour**. Không đưa trực tiếp vào `contourArea`
hoặc `arcLength`. Module đo hiện tại fit trực tiếp tập điểm không thứ tự sau khi loại biên lỗ bên trong.
Chế độ `none` vẫn ghi báo cáo rỗng hiện hành, tránh giữ lại kích thước của lần chạy trước.

## Đo kích thước, dung sai và cấu hình dùng lại

- `--measurement rectangle`: fit 4 cạnh vuông góc, trả `length` và `width`; báo lỗi khi thiếu cạnh/biên không phù hợp.
- `--measurement circle`: fit đường tròn ngoài, trả `diameter`; không phải đường kính lỗ.
- `--measurement bounding-box`: chiều dài/rộng bao ngoài, phù hợp chìa khóa hoặc vật không đều; không phải kích thước theo chuẩn định vị của bản vẽ.
- `--measurement none`: chỉ trích biên và tọa độ như trước. Mặc định là `rectangle`.

Fit hình học diễn ra sau khi đổi từng điểm sang mm; không lấy bounding box pixel rồi nhân một hệ số chung.
Một ROI đo một vật đầy đủ. Bóng, biên thiếu, vật cao lệch mặt phẳng chuẩn có thể làm phép đo sai.

Ví dụ **minh họa**, chỉ dùng số danh nghĩa/dung sai phù hợp bản vẽ của bạn:

```powershell
python -m src.pipeline `
  --image object_undistorted.png --already-undistorted `
  --homography-mode reuse --homography-file outputs/homography.json `
  --select-roi --measurement bounding-box `
  --tolerance length 60 -0.5 0.5 --tolerance width 25 -0.3 0.3 `
  --output outputs/check_tolerance
```

`--tolerance NAME NOMINAL LOWER_DEV UPPER_DEV` dùng sai lệch có dấu, đơn vị mm.
Chưa xác nhận ngân sách độ không đảm bảo thì kết luận là `NOT_VALIDATED`.
Không dùng RMSE bàn cờ thay cho độ không đảm bảo phép đo.

Sau khi thực sự đánh giá bằng mẫu chuẩn, có thể bổ sung các cờ:
`--standard-uncertainty-mm <u_mm> --relative-standard-uncertainty <u_relative> --coverage-factor 2 --uncertainty-validated`.
Hai số `u` là độ không đảm bảo **chuẩn** độc lập, không phải dung sai hoặc sai số cực đại;
`--uncertainty-validated` là xác nhận của người dùng, không tự chứng nhận hệ thống.
Các cờ này không được tự đặt bằng số đoán. Xem công thức và guard band trong [tài liệu API](docs/inspection-api.md).

Dùng [cấu hình rectangle](config/rectangle.example.json), [circle](config/circle.example.json)
hoặc [bounding-box](config/bounding-box.example.json) qua `--config`;
`--save-config` lưu các lựa chọn thực tế, kể cả ROI. Cờ CLI được chỉ định sẽ ghi đè trường tương ứng từ config.

```powershell
python -m src.pipeline `
  --image object_undistorted.png --already-undistorted `
  --homography-file outputs/homography.json `
  --config outputs/key_measurement/inspection_config.json `
  --output outputs/key_repeat
```

Đánh giá nhiều lần chụp độc lập của **cùng mẫu chuẩn**, thay số tham chiếu bằng số thực:

```powershell
python -m src.validation `
  --reports outputs/run01/measurements.json outputs/run02/measurements.json outputs/run03/measurements.json `
  --reference length 60 --reference width 25 `
  --output outputs/reference_validation.json
```

Báo cáo gồm độ lặp lại, bias, RMSE và sai số tuyệt đối lớn nhất; không tự cập nhật H, bù sai số hay bật trạng thái validated.
Không dùng nhiều lần chạy lại cùng một ảnh để kết luận độ lặp lại camera.

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
   Dời seed sang lân cận có gradient mạnh hơn trên trục gradient chính, loại trùng;
   tránh mất cả một cạnh do vị trí seed Canny lệch cực đại gradient.
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
python -m unittest discover -s tests -p "test_*.py" -v
```

Kiểm tra gồm polarity/ROI, HSV qua mốc hue, nhiễu/lỗ/vật chạm biên, biên nghiêng và biên cong
có vị trí biết trước, mask không tạo biên giả, calibration NPZ/PKL, kích thước/crop,
homography, flow bàn cờ tổng hợp + vật và xuất file, hồi quy lưu ảnh camera sạch.
Webcam thật và độ chính xác kích thước sản phẩm chưa được kiểm chứng bằng bộ test này.
Các test mới kiểm tra hình học quay/phối cảnh, nhiễu/ngoại lai, hình thiếu/sai loại, loại biên lỗ,
QC có guard band, API không mở cửa sổ/ghi file, tiến độ/hủy, cấu hình CLI và báo cáo không giữ kết quả cũ.

## Tài liệu API tham khảo

- [OpenCV thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [OpenCV Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [OpenCV calibration và homography](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
