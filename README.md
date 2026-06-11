# 🛡️ Hệ thống Phát hiện & Quản trị Rủi ro Giao dịch Gian lận (Streamlit Web App)

Ứng dụng web này được chuyển đổi tự động từ tài liệu nghiên cứu và mô hình hóa trong file Jupyter Notebook `phat_hien_giao_dich_gian_lan.ipynb`. Hệ thống cho phép người dùng nạp dữ liệu, cấu hình tham số, huấn luyện các thuật toán học máy phổ biến để phát hiện hành vi gian lận tài chính một cách trực quan, chính xác.

## 📌 Các Tính Năng Chính
- **Cấu hình động nâng cao (Sidebar):** Thay đổi linh hoạt 3 kiến trúc mô hình (`Random Forest`, `Decision Tree`, `Logistic Regression`). Cho phép tinh chỉnh siêu tham số chuyên sâu và kích hoạt kỹ thuật cân bằng lớp mất cân bằng (`SMOTE`).
- **Phân tích tổng quan dữ liệu (Tab 1):** Cung cấp góc nhìn thống kê mô tả chi tiết của 14 biến đặc trưng (`X_1` -> `X_14`).
- **Trực quan hóa dữ liệu tương tác (Tab 2):** Sử dụng thư viện `Plotly` vẽ phân phối tần suất của biến mục tiêu `default` và các đặc trưng đầu vào.
- **Đánh giá hiệu năng mô hình (Tab 3):** Tái hiện đầy đủ các chỉ số đánh giá chuyên sâu bao gồm `Accuracy`, `Precision`, `Recall`, `F1-Score`, Ma trận nhầm lẫn trực quan bằng bản đồ nhiệt (Heatmap) và độ quan trọng của đặc trưng (Feature Importance).
- **Hệ thống Dự báo Rủi ro (Tab 4):** Hỗ trợ 2 phương thức dự báo linh hoạt: nhập liệu thông số đơn lẻ hoặc xử lý tệp danh sách dữ liệu kiểm thử hàng loạt (Batch Processing) cho phép xuất báo cáo dạng file `.csv`.

## 📂 Yêu cầu Cấu trúc File Dữ liệu Đầu vào (Schema)
File dữ liệu huấn luyện mẫu (ví dụ: `dataset1.csv`) tải lên hệ thống bắt buộc phải chứa các cột thông tin có định dạng số cụ thể sau:
- Các biến đặc trưng độc lập: **`X_1`, `X_2`, `X_3`, `X_4`, `X_5`, `X_6`, `X_7`, `X_8`, `X_9`, `X_10`, `X_11`, `X_12`, `X_13`, `X_14`**.
- Biến mục tiêu phân loại nhị phân: **`default`** (Nhận giá trị `0` ứng với giao dịch bình thường hoặc `1` ứng với giao dịch rủi ro gian lận).

## 🚀 Hướng Dẫn Cài Đặt và Khởi Chạy Chạy Ứng Dụng

### Bước 1: Khởi tạo và thiết lập môi trường ảo (Khuyên dùng)
```bash
# Tạo môi trường ảo python mới
python -m venv venv

# Kích hoạt môi trường ảo (Hệ điều hành Windows)
.\venv\Scripts\activate

# Kích hoạt môi trường ảo (Hệ điều hành macOS/Linux)
source venv/bin/activate
