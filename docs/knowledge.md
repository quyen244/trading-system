# Knowledge 

## Backtesting 

### PnL

**PnL là gì?**

**PnL = Profit and Loss** → **Lãi & Lỗ**
Dùng để đo **kết quả tài chính** của một giao dịch, một chiến lược, hoặc cả danh mục.

---

## 1️⃣ PnL cơ bản

### 📈 Với 1 lệnh trade

[
\text{PnL} = (\text{Giá bán} - \text{Giá mua}) \times \text{Khối lượng}
]

**Ví dụ**

* Buy BTC: 40,000
* Sell BTC: 41,000
* Size: 0.1 BTC

👉 PnL = (41,000 − 40,000) × 0.1 = **+100 USD**

---

## 2️⃣ Các loại PnL quan trọng

### 🔹 Realized PnL (PnL đã chốt)

* Lãi/lỗ **sau khi đóng lệnh**
* Đã “ăn tiền” hoặc “mất tiền” thật

### 🔹 Unrealized PnL (PnL tạm tính)

* Lãi/lỗ **khi lệnh còn mở**
* Phụ thuộc giá thị trường hiện tại

---

## 3️⃣ Gross vs Net PnL

* **Gross PnL**: chưa trừ phí
* **Net PnL**: đã trừ

  * phí giao dịch
  * funding fee
  * commission

👉 Trong hệ thống trading **luôn dùng Net PnL**

---

## 4️⃣ PnL trong ML / Trading System (liên quan dự án bạn hay làm)

PnL thường dùng để:

* Đánh giá **strategy performance**
* So sánh model (Model A vs Model B)
* Là input cho:

  * Sharpe Ratio
  * Max Drawdown
  * Calmar Ratio

Ví dụ:

```text
Model accuracy cao ❌
Nhưng PnL âm ❌  → model vô dụng
```

---

## 5️⃣ PnL ≠ Return

| Khái niệm | Ý nghĩa                     |
| --------- | --------------------------- |
| PnL       | Lãi/lỗ tuyệt đối (USD, VND) |
| Return    | Lãi/lỗ theo %               |
| ROI       | Hiệu quả trên vốn           |
| Sharpe    | Lãi / rủi ro                |

---

## 6️⃣ Câu nói dân trade hay dùng 😄

* “PnL xanh” → đang lãi
* “PnL đỏ” → đang lỗ
* “Giữ PnL trước đã, tối ưu sau”

---

Nếu bạn muốn, mình có thể:

* Giải thích **PnL trong backtest**
* Cách log **PnL vào MLflow**
* Liên hệ PnL với **Sharpe / drawdown**
* Viết **code Python tính PnL** cho trading system của bạn
