# Phương án 2: Triple Barrier Method (Phương pháp 3 rào chắn) - Khuyên dùng

## Đây là phương pháp chuẩn mực trong các quỹ định lượng (được đề xuất bởi Marcos Lopez de Prado), giải quyết triệt để câu hỏi "bao nhiêu cây nến".

Thay vì hỏi "giá ở cây nến thứ 3 là bao nhiêu", chúng ta hỏi: "Trong vòng 10 cây nến tới, giá chạm Cắt lỗ (SL) trước hay chốt lời (TP) trước, hay không chạm gì cả?"Bài toán: Dự báo xác suất chạm các ngưỡng.

    Thiết lập 3 rào chắn:

    Rào trên (Upper Barrier): Mức chốt lời (VD: +2% hoặc +2*ATR).

    Rào dưới (Lower Barrier): Mức cắt lỗ (VD: -1% hoặc -1*ATR).

    Rào dọc (Vertical Barrier): Giới hạn thời gian (VD: sau 12 cây nến).

Đầu ra (Label - $Y$):

    Class 1 (Buy Signal): Nếu giá chạm Rào trên trước tiên.

    Class -1 (Sell Signal): Nếu giá chạm Rào dưới trước tiên.

    Class 0 (No Trade): Nếu chạm Rào dọc (hết giờ mà giá chưa chạy đủ mạnh).

Ưu điểm: Phương pháp này mô phỏng thực tế trading nhất vì nó bao gồm cả quản lý rủi ro (Stoploss) và thời gian nắm giữ lệnh.

# Phương án 3: Meta-Labeling (Dán nhãn phụ)

Thay vì bắt AI tự tìm điểm vào lệnh, bạn dùng một chỉ báo kỹ thuật cơ bản (ví dụ: RSI < 30 hoặc MA Cross) để tìm điểm vào tiềm năng, sau đó dùng AI để lọc tín hiệu đó.

Bài toán: Phân loại tín hiệu này là ĐÚNG hay SAI.

Quy trình:Hệ thống cơ bản báo: "Cắt lên MA20 -> Mua".

Đưa dữ liệu vào GRU + XGBoost.

Mô hình trả lời câu hỏi: "Nếu mua ở đây, xác suất có lãi là bao nhiêu?"

Đầu ra (Label - $Y$):

1 (Trade): Nếu tín hiệu đó dẫn đến lợi nhuận dương.

0 (Skip): Nếu tín hiệu đó dẫn đến thua lỗ.