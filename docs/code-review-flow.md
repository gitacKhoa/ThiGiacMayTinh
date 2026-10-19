# Rà soát code và nối flow segmentation/sub-pixel

Ngày: 2026-10-20. Phạm vi: toàn bộ file Python trong [src](../src) và [tests](../tests),
cấu hình, dependencies và điểm nối dữ liệu. Tài liệu Word có thay đổi sẵn của người dùng
được giữ nguyên. Không chạy lại calibration để ghi đè ma trận/ảnh đã có.

## Phát hiện và xử lý

| Mức ảnh hưởng | Phát hiện | Xử lý |
| --- | --- | --- |
| Cao | [Capture calibration](../src/calibration/camera-calibration/capture_calibration_images.py) vẽ trực tiếp vào frame rồi lưu frame đó; ảnh cũ `calibration_00.jpg` có chữ Captured | Lưu bản sao frame trước khi vẽ; test hồi quy so sánh pixel đã lưu với frame gốc |
| Cao | Capture dùng 9×6 góc trong, calibration dùng 8×5 trên bàn cờ hiện có | Thống nhất capture thành 8×5; homography tiếp tục cho truyền cols/rows riêng |
| Cao | Nhiều bản sao undistort: alpha/crop khác nhau; live còn resize ảnh đã crop | Dùng chung [undistort](../src/calibration/undistort.py); main flow alpha=0/crop một lần; live map theo camera matrix đã dịch principal point; không resize |
| Cao | Không lưu độ phân giải calibration, không phát hiện khi áp ma trận cho ảnh kích thước khác | Calibration mới lưu image_size; loader giữ metadata; kiểm tra kích thước; pipeline/live yêu cầu kích thước cho file cũ |
| Cao | [Segmentation](../src/segmentation/segment.py) và [sub-pixel](../src/edge_detection/subpixel_edge.py) là file trống | Đã triển khai và nối vào [pipeline](../src/pipeline.py) |
| Vừa | Loader NPZ chỉ chấp nhận mtx/dist trong khi PKL dùng cả alias; Path object không có endswith | Hỗ trợ Path, cả hai bộ tên khóa, đóng NPZ, kiểm tra ma trận/hệ số hữu hạn |
| Vừa | Calibration dùng kích thước ảnh đọc cuối cùng; không kiểm tra ảnh lỗi/khác kích thước | Từ chối ảnh không đọc được và tập ảnh trộn độ phân giải |
| Vừa | `calculate_reprojection_error` chia chuẩn L2 cho N, không phải sqrt(N) | Sửa thành RMS theo từng view; hàm này chưa nằm trong main calibration |
| Vừa | Ảnh chụp có thể bị ghi đè khi chạy lại; bỏ qua imwrite=False | Hai script capture chọn tên chưa tồn tại, báo lỗi lưu; webcam tạo thư mục cha |
| Vừa | Capture calibration lưu vào đường dẫn phụ thuộc working directory | Định vị thư mục data theo vị trí file |
| Vừa | Homography chưa kiểm tra hình học suy biến và chưa có hàm chuyển point cloud | Kiểm tra thông số, ma trận và điểm ở vô cực; thêm points_to_mm float64 |
| Nhỏ | Test homography dùng os.getcwd để import, không reset điểm click | Import theo vị trí file; reset điểm; kiểm tra true_mm dương |
| Nhỏ | Script verify tự lặp loader/undistort và bỏ qua kết quả ghi ảnh | Dùng module chung; tạo thư mục và kiểm tra lưu ảnh |

`verify_calibration` vẫn cho chọn `--crop`: đây là công cụ xem so sánh, không phải ảnh có
thể hoán đổi tùy ý với ảnh pipeline. Chỉ dùng homography và điểm trên cùng một ảnh,
cùng crop/alpha/độ phân giải. Giá trị mặc định của hàm undistort dùng chung vẫn là crop=True.

## Kiểm chứng đã thực hiện

- Môi trường: Python 3.11.9, OpenCV 4.11.0, NumPy 2.2.4.
- **18/18 test tự động đạt** trong [test_vision_pipeline.py](../tests/test_vision_pipeline.py).
  Kiểm tra cú pháp toàn bộ src/tests bằng compileall đạt; editor không báo lỗi.
- Biên nghiêng tổng hợp với pháp tuyến (0,8; 0,6), pha 100,3 pixel:
  234 điểm hợp lệ, MAE trước tinh chỉnh 0,300 pixel, sau tinh chỉnh 0,06214 pixel.
  Đây là ảnh tổng hợp trơn; không đại diện toàn bộ điều kiện camera thực tế.
- Test tích hợp tạo bàn cờ có ô 30 pixel tương ứng 15 mm, vật 100×120 pixel:
  point cloud sau homography đạt span 50×60 mm trong dung sai test 0,25 mm.
- Ảnh thật [homography_test.jpg](../data/calibration_test/homography_test.jpg): dùng HSV,
  ROI và lệnh mẫu trong [README](../README.md); 1.226 điểm hợp lệ/1.294 ứng viên,
  RMSE khớp bàn cờ 0,1314559 mm với square_mm=29.
- Đã xem trực quan mask và overlay: vùng vật xanh được chọn; cạnh trên còn đoạn thiếu,
  biên trái có nhiễu, cạnh dưới có ảnh hưởng bóng. Không đánh đồng kết quả này với
  contour đo kích thước hoàn chỉnh hoặc đạt dung sai.
- 15 ảnh calibration hiện tại có cùng kích thước 1280×720. File PKL cũ giữ nguyên;
  không tự thêm metadata giả định vào kết quả calibration đã có.

## Phần chưa hoàn thiện / cần thông tin trước khi triển khai tiếp

1. [measure.py](../src/measurement/measure.py) còn trống. Cần xác định đại lượng:
   khoảng cách hai mép, đường kính lỗ, chiều rộng/chiều dài theo datum hay bounding box.
   Không tự suy diễn phép đo từ point cloud chưa có thứ tự.
2. [uncertainty.py](../src/measurement/uncertainty.py) và
   [tolerance_check.py](../src/qc/tolerance_check.py) còn trống; chưa có nominal/tolerance,
   số phép đo lặp và ground truth. Không xuất quyết định PASS/FAIL giả.
3. [GUI](../src/gui/app.py), [config](../config/config.yaml), [ground truth](../data/ground_truth.csv)
   hiện trống. Tham số pipeline được truyền rõ qua CLI/API, chưa có loader YAML.
4. Homography yêu cầu mặt đo và bàn cờ đồng phẳng. Độ dày vật, độ phẳng giấy in,
   bóng đổ, focus, ánh sáng và cạnh ô thực tế cần kiểm chứng bằng phép đo vật chuẩn.
5. Giảm ngưỡng Canny giúp nhận cạnh yếu nhưng cũng nhận bóng/họa tiết. Cần nền và
   chiếu sáng phù hợp, hoặc chọn kênh đặc trưng theo vật nếu grayscale thiếu tương phản.
6. Các script camera đã được kiểm tra bằng mock/hồi quy; chưa mở webcam thật trong phiên này.
   Ảnh calibration cũ có annotation; bản sửa chỉ cải thiện lần chụp sau. Cần bộ ảnh sạch
   nếu muốn đánh giá lại chất lượng calibration, không kết luận ma trận cũ vô hiệu chỉ vì có chữ ngoài bàn cờ.

## Hợp đồng dữ liệu giữa các module

- Ảnh đầu vào segmentation/sub-pixel: uint8 gray/BGR/BGRA; main CLI dùng BGR.
- Mask đầu ra: uint8 0/255 toàn ảnh. ROI không đổi gốc tọa độ đầu ra.
- Sub-pixel: float64 `(N, 2)` thứ tự `(x, y)`; tập điểm không có thứ tự contour.
- Gradient được tính từ ảnh thật đã làm mượt; morphology chỉ ảnh hưởng mask chọn vùng.
- Homography: pixel→mm, tính trên cùng ảnh undistort/crop; không transform từ ảnh raw.
- Metadata lưu source size, crop ROI và camera matrix mới khi pipeline thực hiện undistort.
- Đầu ra mm là tọa độ trên mặt phẳng chuẩn, chưa phải đại lượng đo hoặc uncertainty.
