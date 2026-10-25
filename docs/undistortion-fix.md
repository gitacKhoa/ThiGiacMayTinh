# Sửa kéo giãn undistort và chuyển sang hệ tọa độ mới

## Nguyên nhân đã kiểm chứng

Code cũ không resize sau crop. Nguồn kéo giãn là nhánh mặc định
`cv2.getOptimalNewCameraMatrix(..., centerPrincipalPoint=False)`, tối ưu hai focal length độc lập.
Với calibration hiện tại, ảnh RAW 1280×720 và `alpha=0`:

| Thông số | Calibration gốc | Đầu ra cũ | Đầu ra mới |
| --- | ---: | ---: | ---: |
| fx | 980,879852 | 784,705318 | 1033,10192 |
| fy | 977,853590 | 922,552439 | 1029,91454 |
| fx/fy | 1,003095 | 0,850581 | 1,003095 |
| Hệ số phóng ngang | — | 0,800001 | 1,053240 |
| Hệ số phóng dọc | — | 0,943446 | 1,053240 |

Như vậy nhánh cũ làm chiều dọc tương đối lớn hơn khoảng 17,93% so với chiều ngang.
Đã bật `centerPrincipalPoint=True`: dùng một hệ số chung, giữ `fx/fy` gốc, sau đó chỉ crop
và trừ offset khỏi principal point. Không ép fx=fy tùy ý và không resize ảnh để ô bàn cờ trông vuông.
`crop=False` cũng dùng nhánh giữ tỷ lệ, với alpha=1. Crop/zoom thay đổi trường nhìn nhưng không kéo riêng một trục.

Undistort chỉ sửa méo ống kính. Một bàn cờ nhìn nghiêng còn phối cảnh; muốn nhìn từ trên xuống
cần một phép rectification mặt phẳng riêng. Đo mm hiện tại biến đổi từng điểm bằng H,
không yêu cầu các ô phải trông vuông trong ảnh undistort.

## Phần được đồng bộ

- [Undistort chung](../src/calibration/undistort.py) được dùng bởi webcam, pipeline, live preview,
  verify, xuất ảnh calibration và công cụ homography thủ công.
- [Sidecar ảnh](../src/calibration/image_geometry.py) giữ matrix gốc/đầu ra, model,
  crop, distortion, kích thước và SHA-256 của file ảnh. File có tên `image.png.geometry.json`.
- [Homography](../src/measurement/homography_profile.py) kiểm tra model và geometry,
  không chỉ kiểm tra chiều rộng/cao. H cũ không được âm thầm dùng với ảnh raw xử lý bằng nhánh mới.
- [Pipeline](../src/pipeline.py) tự nạp sidecar khi `--already-undistorted`; report giữ metadata tương ứng.
- [Verify](../src/verify_calibration.py) ghép ảnh ở pixel scale gốc, chỉ thêm padding;
  có thể xuất ảnh sạch riêng qua `--undistorted-output`.
- Đo hình học vẫn ở hệ mm, overlay ánh xạ bằng inverse H; không sửa bằng hệ số mm/pixel đoán.

## Cách chạy lại đúng sau bản sửa

Chạy từ thư mục gốc dự án. Ví dụ dưới dùng tên mới để không ghi đè bộ ảnh/H cũ.
29 mm phải là kích thước ô đã đo thực, 8×5 là số góc **trong**.

### 1. Đặt bàn cờ trên mặt phẳng đo, tính H mới

```powershell
python tests/capture_webcam.py `
  --calib outputs/calibration_results/calibration_data.pkl `
  --calibration-size 1280 720 --width 1280 --height 720 `
  --homography-mode compute --square-mm 29 --cols 8 --rows 5 `
  --homography-file outputs/homography_v2.json `
  --output board_undistorted_v2.png
```

Space/C chụp và tính H; Q thoát. Không thấy bàn cờ thì báo lỗi, không ghi H mới.
Giữ nguyên camera, focus/zoom, độ phân giải, mặt phẳng đo sau bước này.

### 2. Bỏ bàn cờ, đặt vật trên đúng mặt phẳng, chụp ảnh vật

```powershell
python tests/capture_webcam.py `
  --calib outputs/calibration_results/calibration_data.pkl `
  --calibration-size 1280 720 --width 1280 --height 720 `
  --homography-mode reuse --homography-file outputs/homography_v2.json `
  --output object_undistorted_v2.png
```

Giữ file `.geometry.json` cạnh từng ảnh. Nếu tên ảnh đã tồn tại, công cụ thêm hậu tố số;
dùng đúng đường dẫn được in ra cho bước tiếp theo. Không đổi tên ảnh mà bỏ quên sidecar.

### 3. Đo vật, chọn lại ROI

```powershell
python -m src.pipeline `
  --image object_undistorted_v2.png --already-undistorted `
  --homography-mode reuse --homography-file outputs/homography_v2.json `
  --select-roi --measurement bounding-box `
  --output outputs/key_v2 --save-config outputs/key_v2/inspection_config.json
```

`bounding-box` dành cho kích thước bao ngoài chìa khóa. Chọn `rectangle`/`circle` khi phù hợp hình học thực.
ROI cũ `(492,295,256,153)` không còn là ROI của ảnh mới. Không giữ config ROI cũ mà chưa kiểm tra lại.
Kích thước vật chỉ có giá trị đo trên mặt phẳng đã hiệu chuẩn; chưa có dung sai/mẫu chuẩn thì chưa xác nhận QC.

### Nếu còn ảnh RAW thay vì chụp lại

```powershell
python -m src.verify_calibration `
  --image raw_board.png `
  --calib outputs/calibration_results/calibration_data.pkl --calibration-size 1280 720 `
  --crop --output outputs/board_comparison.png `
  --undistorted-output board_undistorted_v2.png
```

Ảnh comparison có nhãn/padding chỉ để xem, **không phải đầu vào phép đo**.
Ảnh `--undistorted-output` sạch và có sidecar, dùng để tính H; vật phải được xử lý cùng calibration/crop.
CLI pipeline chế độ compute vẫn cần ROI có vật; muốn tính riêng H từ ảnh bàn cờ, dùng API
`create_homography_profile` với ảnh sạch và geometry đã nạp, rồi `save_homography_profile`.

## Ảnh cũ và cơ chế chặn nhầm

Không sửa hoặc xóa `board_undistorted.png`, `object_undistorted.png`, `outputs/homography.json` cũ.
Không undistort chúng lần hai vì chúng đã qua sửa méo. Không thể khôi phục các pixel bị crop mất từ một ảnh đã lưu.
Mặc định ảnh không có sidecar sẽ bị từ chối; đây là lỗi có chủ đích để tránh trộn hai hệ tọa độ.

`--allow-unverified-geometry` chỉ dùng khi tự xác minh cặp ảnh/H ngoài khớp nhau.
Nó không sửa kéo giãn, không cập nhật H, không bỏ qua checksum hỏng hoặc geometry mismatch đã biết.
Không nên dùng cờ này để tiếp tục đo ảnh mới bằng H cũ. Nếu không có metadata, phần mềm không thể
tự chứng minh hai ảnh cùng calibration, crop hoặc camera pose.

H cũ vẫn có thể quy đổi đúng ảnh **cũ tương ứng** dù hình hiển thị bị kéo: phép biến đổi hai trục
có thể được H bù lại. Nhưng không được dùng H đó cho ảnh mới và không được lấy sự khớp này làm chứng nhận độ chính xác.

## Kiểm chứng trong lần sửa này

- Test tái hiện thất bại trước sửa vì hệ số ngang/dọc khác nhau ở cả crop=True và crop=False.
- Sau sửa: giữ tỷ lệ focal, bàn cờ tổng hợp vuông vẫn vuông khi nhìn chính diện, live remap/crop cùng tọa độ.
- Kiểm tra ảnh/H khác model nhưng cùng kích thước bị từ chối, sidecar stale/hash sai bị từ chối,
  và workflow ảnh lưu → nạp H → đo mm → xuất lại ảnh/geometry chạy được.
- Đã chạy ảnh RAW calibration thực qua verify trong thư mục `outputs/undistort_fix`.
- `board_projection_preview.png` trong thư mục đó là **preview chẩn đoán** từ ảnh bàn cờ cũ:
  chỉ đổi phép chiếu `K_new @ inverse(K_old)`, không undistort ống kính lần hai, không sinh H/sidecar để đo.
  Dùng alpha=1 để nhìn rộng hơn; vùng đen là vùng không có dữ liệu trong ảnh cũ.
  Tỷ số bước ô trung bình cao/rộng giảm từ 1,173 xuống 0,995. Đây không phải chứng nhận sai số đo.
- Chưa vận hành webcam thực hay đối chiếu thước chuẩn trong lần sửa này.

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
