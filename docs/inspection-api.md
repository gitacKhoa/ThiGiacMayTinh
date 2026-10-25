# API đo và nền tảng cho giao diện

## Phạm vi hiện tại

Pipeline đã nối từ chuẩn bị ảnh đến đo mm, đánh giá dung sai và báo cáo.
GUI chưa triển khai; camera, chọn ROI, lưu file và hiển thị là trách nhiệm của adapter.
Lõi xử lý không phụ thuộc PyQt/Tkinter/web, không gọi input/print/cửa sổ và không tự ghi file.

| Thành phần | Trách nhiệm |
| --- | --- |
| [inspection.py](../src/inspection.py) | API chung: chuẩn bị ảnh, điều phối đo, callback tiến độ/hủy, kết quả có cấu trúc |
| [processing.py](../src/processing.py) | Segmentation, subpixel, biến đổi điểm sang mm; API cũ còn được re-export qua pipeline |
| [measure.py](../src/measurement/measure.py) | Thuật toán hình học độc lập ảnh/camera/filesystem |
| [uncertainty.py](../src/measurement/uncertainty.py) | Thành phần độ không đảm bảo, RSS và thống kê lặp lại |
| [tolerance_check.py](../src/qc/tolerance_check.py) | Giới hạn có sai lệch bất đối xứng và quyết định có guard band |
| [configuration.py](../src/configuration.py) | Cấu hình JSON version 1 dùng chung CLI/GUI |
| [reporting.py](../src/reporting.py) | Chuyển kết quả sang dict, vẽ ảnh trong bộ nhớ hoặc xuất file khi được yêu cầu |
| [pipeline.py](../src/pipeline.py) | Adapter CLI; đọc ảnh/calibration/H, chọn ROI tùy chọn, gọi service, in/lưu kết quả |
| [validation.py](../src/validation.py) | Đối chiếu nhiều báo cáo với kích thước mẫu chuẩn do người dùng cung cấp |

## Ví dụ cho GUI không mở cửa sổ OpenCV

```python
import cv2
from src.configuration import load_config
from src.calibration.image_geometry import load_image_geometry
from src.inspection import inspect_image, prepare_image, InspectionCancelled
from src.measurement.homography_profile import load_homography_profile, homography_for_image
from src.reporting import inspection_to_dict, render_inspection, save_inspection

# I/O do adapter quản lý. Ảnh ví dụ đã được undistort.
input_image = cv2.imread("object_undistorted_v2.png")
prepared = prepare_image(input_image)
prepared.preprocessing = load_image_geometry("object_undistorted_v2.png", input_image)
profile = load_homography_profile("outputs/homography_v2.json")
H = homography_for_image(profile, prepared.image, preprocessing=prepared.preprocessing)
config = load_config("outputs/key_measurement/inspection_config.json")

# Chạy trong worker; GUI truyền callback phát signal/đẩy queue nếu cần.
result = inspect_image(prepared.image, H, config)
payload = inspection_to_dict(result)       # dict có thể JSON hóa, để hiển thị bảng
preview_bgr = render_inspection(prepared.image, result)  # ndarray, không imshow
# Chỉ gọi khi người dùng bấm Lưu:
save_inspection("outputs/gui_run", prepared.image, result, config=config,
                metadata={"homography_reference": profile, "preprocessing": prepared.preprocessing})
```

Nếu ảnh gốc: nạp calibration qua `load_calibration_data`, rồi truyền dictionary vào
`prepare_image(raw, calibration=calibration, calibration_size=(1280, 720))`.
Kích thước phải là kích thước calibration thực; có `image_size` trong calibration thì không cần ghi lại.
Không undistort lần hai. Không tự resize/crop ảnh sau khi chuẩn bị.
Muốn tính H mới, gọi `create_homography_profile(prepared.image, square_mm, pattern_size,
preprocessing=prepared.preprocessing)` rồi lưu profile qua adapter; không thấy bàn cờ sẽ báo lỗi.
Module homography không đọc `.pkl`.
Ảnh lưu mới có sidecar `.geometry.json` chứa hash và hình học. Chỉ đọc sidecar ở adapter;
API thuần không tự đọc file. `homography_for_image` mặc định từ chối metadata thiếu/không khớp.
`allow_unverified=True` chỉ dành cho cặp ảnh/H ngoài đã tự xác minh, không bỏ qua mismatch đã biết.

### Hợp đồng cho UI

- Đầu vào ảnh uint8 BGR hoặc gray đã undistort; ROI `(x,y,w,h)` trong hệ tọa độ ảnh này.
  Nếu preview UI bị thu nhỏ/letterbox, phải đổi tọa độ chuột về ảnh nguồn trước khi tạo ROI.
- API nhận ma trận H đã kiểm tra tương thích bằng `homography_for_image`, không nhận đường dẫn H.
- Mỗi lần chạy có `InspectionConfig` riêng, copy frame và config khi đưa vào worker.
  Không sửa chúng trong lúc tác vụ đang xử lý.
- `on_progress(stage, fraction)` có các stage `processing`, `measurement`, `uncertainty`, `tolerance`, `complete`.
  Fraction là tiến độ theo giai đoạn, không phải ước lượng thời gian còn lại.
- `is_cancelled()` trả bool; service ném `InspectionCancelled` giữa các stage.
  Không ngắt tức thời ở bên trong một lời gọi OpenCV đang chạy.
- Callback chạy trong thread gọi service: GUI phải dùng signal/queue, không sửa widget trực tiếp từ worker.
- `ValueError` chỉ đầu vào/hình học không phù hợp; adapter có thể bắt thêm `cv2.error`, `OSError`
  ở lớp tương ứng và hiển thị thông báo. Không biến lỗi thành kết quả PASS hoặc kích thước 0.
- `InspectionResult` giữ cả processing (mask/điểm/H), measurement, uncertainties, tolerance_results,
  qc_status, warnings và measurement_point_indices (chỉ số điểm biên ngoài đưa vào phép đo).
- Ảnh overlay giữ nguyên pixel gốc; bảng chữ nằm ở panel riêng. Đổi BGR→RGB ở adapter khi framework yêu cầu.

## Các định nghĩa kích thước

- `rectangle`: fit 4 cạnh thẳng vuông góc trong hệ mm, `length >= width`.
  Có chống ngoại lai, kiểm tra RMSE, tỷ lệ inlier và độ phủ trên từng cạnh.
  Đây là mô hình chữ nhật danh nghĩa, không đo riêng độ vuông góc hay độ thẳng.
- `circle`: fit hình học đường tròn ngoài, trả `diameter`; yêu cầu độ phủ góc đủ lớn.
  Không đo đường kính lỗ bên trong, độ tròn hay ren.
- `bounding-box`: hình chữ nhật bao diện tích nhỏ nhất của các điểm biên ngoài đã trích trong hệ mm.
  Dành cho kích thước tổng thể của vật không đều; mọi điểm đưa vào đều được giữ, không cắt ngoại lai
  vì có thể chính là phần nhô thật. Nhiễu/bóng/biên thiếu có thể làm thay đổi kích thước bao.
  Không bảo đảm trục đo trùng chuẩn định vị trên bản vẽ; không có covariance fit.
- `none`: chỉ biên/tọa độ; không kết luận dung sai. Ở API thuần có thể không truyền H;
  CLI hiện vẫn yêu cầu H theo workflow compute/reuse cũ.

Fit diễn ra trên từng điểm đã biến đổi pixel→mm, không dùng một hệ số mm/pixel chung.
Các biên lỗ bên trong được loại khỏi phép đo ngoài. Một ROI chỉ đo một vật hoàn chỉnh;
mặc định segmentation giữ thành phần lớn nhất nên người dùng phải kiểm tra mask chọn đúng vật.
Không có tự đoán loại hình hoặc tự suy ra kích thước danh nghĩa/dung sai.
Đo nhiều vật, đặc trưng lỗ, khoảng cách tâm, chuẩn định vị bản vẽ và GD&T là các mở rộng riêng,
không được ngầm coi là đã hỗ trợ.

## Độ không đảm bảo và QC

Với kích thước `L`, ngân sách gồm các thành phần chuẩn độc lập:

```text
u = sqrt(u_fit² + standard_mm² + (L * relative_standard)²)
U = coverage_factor * u
```

`relative_standard=0.001` nghĩa là 0,1%, không phải 0,001 mm.
`u_fit` từ covariance fit, giả định residual độc lập và H chính xác. Nó không mô tả đủ
correlation của pixel, calibration, nội suy, sai lệch hệ thống, độ cao vật, điều kiện ánh sáng.
Các thành phần ngoại sinh phải được xác định bằng thử nghiệm và thông tin mẫu chuẩn;
không cộng trùng một thành phần đã nằm trong ngân sách khác. Bias đáng kể cần xử lý/đặc trưng,
không tự lấy RSS để che một sai lệch chưa kiểm soát. `k=2` không tự chứng minh mức tin cậy 95%.

Bounding-box không có `u_fit`: giá trị 0 trong trường fit chỉ là không có mô hình covariance.
Nếu không có ngân sách ngoại sinh, `available=false`, JSON ghi `standard_mm/expanded_mm=null`,
CSV để trống và UI/CLI ghi unknown; tuyệt đối không diễn giải thành phép đo chính xác tuyệt đối.
API typed result giữ số 0 làm placeholder khi `available=false`; consumer phải kiểm tra cờ này.

Mặc định `validated=false`; chỉ có độ không đảm bảo tạm tính, không tự quyết định đạt/không đạt.
`validated=true` là xác nhận của người dùng sau khi đặc trưng trên mẫu vật lý và phải có ít nhất một
thành phần ngoại sinh dương. Phần mềm không có cách tự kiểm chứng xác nhận này.
RMSE của bàn cờ chỉ là độ khớp các điểm dùng tính H, không được đưa thẳng vào ngân sách.

Giới hạn `low=nominal+lower_deviation`, `high=nominal+upper_deviation`, khoảng đo `[L-U,L+U]`:

| Trạng thái | Điều kiện |
| --- | --- |
| `NOT_CONFIGURED` | Chưa cung cấp dung sai |
| `NOT_VALIDATED` | Có dung sai nhưng ngân sách chưa được người dùng xác nhận |
| `PASS` | Toàn bộ khoảng đo nằm trong giới hạn, kể cả bằng giới hạn |
| `FAIL` | Toàn bộ khoảng đo nằm ngoài giới hạn |
| `INDETERMINATE` | Khoảng đo giao/cắt ranh giới, chưa đủ chắc chắn để kết luận |
| `PARTIAL` | Các kích thước đã cấu hình đều PASS nhưng vẫn thiếu dung sai cho kích thước khác |

QC tổng ưu tiên FAIL, rồi NOT_VALIDATED, INDETERMINATE, PARTIAL, PASS.
Một số đo ở ngoài giới hạn nhưng khoảng độ không đảm bảo còn chạm giới hạn là INDETERMINATE,
không ép thành FAIL. Có thể xem `inside_nominal_limits` để biết riêng vị trí số đo trung tâm.

## Cấu hình và lưu vết

JSON version 1 gồm `roi`, `segmentation`, `edges`, `measurement`, `uncertainty`, `tolerances`.
Tham khảo [rectangle](../config/rectangle.example.json), [circle](../config/circle.example.json),
[bounding-box](../config/bounding-box.example.json). Các ví dụ cố ý không đặt dung sai giả.
Cờ CLI chỉ ghi đè trường được truyền; các trường khác giữ từ config hoặc default.
Config không chứa đường dẫn ảnh, thiết bị camera hay H: adapter quản lý phiên làm việc.
Lệnh và nguồn ảnh/H được ghi vào metadata báo cáo để truy vết.

`save_inspection` ghi đè các tên output chuẩn trong thư mục được chọn; dùng thư mục mới cho mỗi lần đo
nếu cần giữ lịch sử. Đây không phải lưu nhiều file theo transaction; lỗi I/O có thể để lại output một phần,
nên adapter chỉ thông báo thành công sau khi hàm trả về. File config có thể lưu qua `save_config`.

## Nghiệm thu vật lý trước khi dùng kết luận QC

1. Giữ camera/focus/zoom/độ phân giải/undistort/crop và mặt phẳng đo đúng phiên tính H.
2. Đo kích thước ô bàn cờ thực; dùng mẫu có kích thước tham chiếu và độ không đảm bảo đã biết.
3. Chụp độc lập nhiều lần, thử nhiều vị trí/góc trong vùng đo và các điều kiện ánh sáng thực tế.
4. Chạy pipeline đúng loại kích thước, kiểm tra mask/biên/overlay từng lần.
5. Dùng `python -m src.validation` theo [README](../README.md) để tổng hợp độ lặp lại, bias, RMSE,
   sai số cực đại. Độ lệch chuẩn mẫu mô tả một lần đo; `sd/sqrt(n)` chỉ cho trung bình n lần độc lập.
6. Xây dựng ngân sách và quy tắc chấp nhận phù hợp mục đích; chỉ khi đó mới xác nhận validated.

Homography chỉ đúng mặt phẳng đã hiệu chuẩn. Không thể tự phát hiện camera di chuyển hoặc vật
có bề dày khiến biên nằm lệch mặt phẳng. Một ảnh có hình nhìn bị kéo theo trục vẫn có thể quy đổi đúng
nếu H và ảnh dùng đúng cùng hình học, nhưng không được thay cách undistort/crop rồi dùng H cũ.
Hình học mới dùng `centerPrincipalPoint=True`, giữ hệ số phóng hai trục bằng nhau và giữ `fx/fy` gốc.
Không âm thầm resize về 1280×720. H/ROI của thuật toán cũ phải được tính/chọn lại;
xem [quy trình chuyển đổi](undistortion-fix.md). Không lấy metadata của H gán cho ảnh chưa xác minh.
