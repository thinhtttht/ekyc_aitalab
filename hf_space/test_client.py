"""Test Client Script for eKYC REST API.
Can test locally (http://127.0.0.1:7860) or against Hugging Face Spaces URL.

Usage:
  python test_client.py --url http://127.0.0.1:7860 --image path/to/face.jpg
"""
import argparse
import sys
import os
import requests

def main():
    parser = argparse.ArgumentParser(description="Test eKYC API")
    parser.add_argument("--url", default="http://127.0.0.1:7860", help="Base URL of API")
    parser.add_argument("--image", default="", help="Path to test image file")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"🔄 Đang kết nối tới: {base_url}...")

    # 1. Test Health Check
    try:
        res = requests.get(f"{base_url}/", timeout=10)
        print(f"✅ Health Check (Status {res.status_code}):", res.json())
    except Exception as e:
        print(f"❌ Không thể kết nối tới {base_url}: {e}")
        return

    # Nếu không có file ảnh truyền vào, tìm ảnh mẫu trong repo
    img_path = args.image
    if not img_path:
        possible_imgs = [
            os.path.join(os.path.dirname(__file__), "..", "realsense_snapshots", "snapshot_2d_20261019_133025.png"),
            os.path.join(os.path.dirname(__file__), "..", "realsense_snapshots", "snapshot_2d_20261019_133714.png"),
        ]
        img_path = next((p for p in possible_imgs if os.path.exists(p)), "")

    if not img_path or not os.path.exists(img_path):
        print("ℹ️ Vui lòng cung cấp file ảnh chân dung qua tham số --image để kiểm thử đăng ký và xác thực.")
        return

    print(f"\n📸 Đang sử dụng file ảnh: {img_path}")

    # 2. Test Register Face
    print("\n--- 1. TEST ĐĂNG KÝ (REGISTER) ---")
    with open(img_path, "rb") as f:
        files = {"image": ("test_face.png", f, "image/png")}
        data = {
            "user_id": "TEST_STUDENT_001",
            "full_name": "Nguyễn Văn Thử Nghiệm",
            "enforce_oval": "false",
            "require_liveness": "false"  # Để test ảnh tĩnh
        }
        res = requests.post(f"{base_url}/api/v1/register", data=data, files=files)
        print(f"Status: {res.status_code}")
        print("Response:", res.json())

    # 3. Test Verify Face
    print("\n--- 2. TEST XÁC THỰC (VERIFY) ---")
    with open(img_path, "rb") as f:
        files = {"image": ("test_face.png", f, "image/png")}
        data = {
            "user_id": "TEST_STUDENT_001",
            "enforce_oval": "false",
            "require_liveness": "false",
            "threshold": "0.55"
        }
        res = requests.post(f"{base_url}/api/v1/verify", data=data, files=files)
        print(f"Status: {res.status_code}")
        print("Response:", res.json())

    # 4. Test List Users
    print("\n--- 3. TEST DANH SÁCH NGƯỜI DÙNG (USERS) ---")
    res = requests.get(f"{base_url}/api/v1/users")
    print(res.json())

if __name__ == "__main__":
    main()
