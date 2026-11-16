# 🚀 HƯỚNG DẪN TỪ A-Z ĐẨY API LÊN HUGGING FACE SPACES
## DEPLOY_TO_HUGGINGFACE.md

Tài liệu này hướng dẫn bạn từng bước chính xác để đưa toàn bộ thư mục `hf_space` lên Hugging Face Spaces để có một REST API online 24/7.

---

## BƯỚC 1: TẠO SPACE MỚI TRÊN HUGGING FACE

1. Đăng nhập vào tài khoản [Hugging Face](https://huggingface.co/).
2. Truy cập vào trang tạo Space: [https://huggingface.co/new-space](https://huggingface.co/new-space)
3. Điền các thông tin:
   * **Space name:** Đặt tên bất kỳ, ví dụ `ekyc-face-api`
   * **License:** Chọn `MIT` hoặc `Apache 2.0`
   * **Select the Space SDK:** Chọn **Docker** *(BẮT BUỘC CHỌN DOCKER)*
   * **Docker template:** Chọn **Blank**
   * **Space hardware:** Chọn **CPU basic • 2 vCPU • 16 GB RAM • Free** (Miễn phí 100%)
   * **Privacy:** Chọn **Public** (để ai cũng gọi được API) hoặc **Private** (chỉ bạn dùng).
4. Bấm nút **"Create Space"**.

---

## BƯỚC 2: ĐẨY CODE & MODEL TỪ THƯ MỤC `hf_space` LÊN HUGGING FACE

Mở Terminal (hoặc PowerShell) tại máy tính của bạn và thực hiện các lệnh sau:

### 1. Di chuyển vào thư mục `hf_space`:
```bash
cd D:\EKYC\hf_space
```

### 2. Khởi tạo Git và bật Git LFS (bắt buộc để đẩy file model 174MB):
```bash
git init
git lfs install
git lfs track "models/*.onnx"
```

### 3. Thêm toàn bộ file và commit:
```bash
git add .
git commit -m "feat: initial eKYC face recognition docker api"
```

### 4. Kết nối tới Space của bạn và đẩy lên:
*(Thay `<username>` và `<space-name>` bằng tài khoản và tên Space của bạn)*:

```bash
git remote add hf https://huggingface.co/spaces/<username>/<space-name>
git branch -M main
git push -u hf main
```

> **Lưu ý xác thực Hugging Face:**
> Khi Git hỏi mật khẩu (Password), bạn **không dùng mật khẩu đăng nhập thông thường** mà tạo một Access Token:
> Vào [Hugging Face Settings -> Access Tokens](https://huggingface.co/settings/tokens) $\rightarrow$ Bấm **New token** $\rightarrow$ Chọn loại **Write** $\rightarrow$ Copy mã token và dán vào khi Git hỏi mật khẩu.

---

## BƯỚC 3: THEO DÕI BUILD & SỬ DỤNG API

1. Sau khi `git push`, quay lại trang Space của bạn trên trình duyệt.
2. Bạn sẽ thấy trạng thái chuyển từ **Building** $\rightarrow$ **Running**. Quá trình build Docker mất khoảng 2 - 3 phút.
3. Khi chuyển sang **Running**, API của bạn đã chính thức hoạt động!

### Địa chỉ API của bạn:
* **Địa chỉ gốc:** `https://<username>-<space-name>.hf.space`
* **Trang tài liệu Swagger tương tác trực tiếp:** `https://<username>-<space-name>.hf.space/docs`
* **Trang Redoc:** `https://<username>-<space-name>.hf.space/redoc`

---

## BƯỚC 4: KIỂM THỬ API ONLINE

Bạn có thể mở terminal trên máy tính và chạy script kiểm thử tự động kèm theo:

```bash
python test_client.py --url https://<username>-<space-name>.hf.space --image "D:\EKYC\realsense_snapshots\snapshot_2d_20261019_133025.png"
```

Hoặc dùng Postman / cURL để gọi:
```bash
curl -X POST "https://<username>-<space-name>.hf.space/api/v1/register" \
  -F "user_id=SV001" \
  -F "full_name=Nguyen Van A" \
  -F "image=@portrait.jpg"
```
