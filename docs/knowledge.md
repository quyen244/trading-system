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

Nói ngắn gọn:

> **CAGR trả lời câu hỏi:**
> *“Nếu vốn của tôi tăng đều mỗi năm, thì mỗi năm tăng bao nhiêu % để từ vốn đầu → vốn cuối?”*

---

## 1️⃣ Công thức CAGR

[
\text{CAGR} = \left(\frac{V_{\text{final}}}{V_{\text{initial}}}\right)^{\frac{1}{N}} - 1
]

Trong đó:

* (V_{\text{initial}}): vốn ban đầu
* (V_{\text{final}}): vốn cuối cùng
* (N): số **năm**

---

## 2️⃣ Ví dụ cực dễ hiểu

### Ví dụ 1: đầu tư 3 năm

* Vốn đầu: **100 triệu**
* Vốn cuối: **200 triệu**
* Thời gian: **3 năm**

[
\text{CAGR} = (200 / 100)^{1/3} - 1
= 2^{1/3} - 1
≈ 26%
]

👉 Nghĩa là: **mỗi năm lãi đều 26%**

---

## 3️⃣ Vì sao không dùng “tổng lợi nhuận / số năm”?

Vì lợi nhuận **có lãi kép**.

### Ví dụ:

* Năm 1: +50%
* Năm 2: -20%

Tổng = +30% ❌
Nhưng:
[
100 → 150 → 120
]

CAGR:
[
(120/100)^{1/2} - 1 ≈ 9.54%
]

👉 CAGR phản ánh **thực tế hơn**

---

## 4️⃣ CAGR trong backtest trading dùng để làm gì?

Trong hệ thống backtest của bạn, CAGR dùng để:

* So sánh **chiến lược có thời gian khác nhau**
* So sánh **strategy vs benchmark (VNIndex, BTC, SP500)**
* Đánh giá **tăng trưởng dài hạn**

📌 CAGR **không nói gì về rủi ro**

---

## 5️⃣ CAGR cao có luôn tốt không?

❌ KHÔNG

Ví dụ:

* Strategy A: CAGR 30%, Max DD 70%
* Strategy B: CAGR 18%, Max DD 15%

👉 Quỹ chuyên nghiệp chọn **B**

---

## 6️⃣ CAGR vs các chỉ số khác (rất quan trọng)

| Chỉ số       | Trả lời câu hỏi               |
| ------------ | ----------------------------- |
| CAGR         | Mỗi năm lời bao nhiêu         |
| Max Drawdown | Có lúc lỗ nặng nhất bao nhiêu |
| Sharpe       | Lợi nhuận / rủi ro            |
| Calmar       | CAGR / Max DD                 |

📌 **CAGR luôn phải xem cùng Max Drawdown**

---

## 7️⃣ CAGR trong equity curve (code mẫu)

```python
def calculate_cagr(equity: pd.Series, periods_per_year=252):
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    return total_return ** (1 / years) - 1
```

---

## 8️⃣ Hiểu sai thường gặp (cảnh báo)

❌ CAGR 50% = mỗi năm đều 50%
❌ CAGR cao = strategy tốt
❌ CAGR dùng cho short-term trade

---

## 9️⃣ Kết luận ngắn gọn

✔️ CAGR = **tốc độ tăng trưởng kép hàng năm**
✔️ Chuẩn để so sánh chiến lược dài hạn
⚠️ Không đo rủi ro
👉 Luôn xem cùng **Max Drawdown & Sharpe**
---

---

## 📊 Sharpe Ratio là gì?

**Sharpe Ratio** đo lường:

> **Mỗi đơn vị rủi ro bạn nhận vào thì bạn được bao nhiêu lợi nhuận**

Nói ngắn gọn:

* **Lời nhiều mà ít biến động → Sharpe cao**
* **Lời nhiều nhưng rung lắc mạnh → Sharpe thấp**

---

## 🧮 Công thức Sharpe

[
\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}
]

Trong đó:

* (R_p): lợi nhuận của strategy
* (R_f): lãi suất phi rủi ro (thường ≈ 0 trong crypto / backtest)
* (\sigma_p): độ biến động (std) của lợi nhuận

👉 Trong trading system, thường dùng **simplified Sharpe**:
[
\text{Sharpe} = \frac{\text{mean(return)}}{\text{std(return)}}
]

---

## 🧠 Hiểu bằng trực giác

### Ví dụ:

| Strategy | CAGR | Biến động | Sharpe |
| -------- | ---- | --------- | ------ |
| A        | 30%  | Rất mạnh  | 0.8    |
| B        | 18%  | Êm        | 1.6    |

👉 **Quỹ chuyên nghiệp chọn B**, không chọn A

---

## 🏆 Sharpe bao nhiêu là tốt?

| Sharpe  | Đánh giá |
| ------- | -------- |
| < 0     | Tệ       |
| 0 – 1   | Kém      |
| 1 – 1.5 | Ổn       |
| 1.5 – 2 | Tốt      |

> 2 | Rất tốt |
> 3 | ❌ nghi ngờ backtest |

📌 **Sharpe > 3** thường là:

* Look-ahead bias
* Entry tại close
* Overfitting

---

## ⚠️ Hiểu sai rất hay gặp

❌ Sharpe cao = chắc thắng
❌ Sharpe áp dụng cho ít trade
❌ So Sharpe giữa timeframe khác nhau

Sharpe **chỉ có ý nghĩa khi**:

* Số trade đủ lớn
* Timeframe giống nhau
* Không có bias

---

## 🔍 Sharpe trong backtest của bạn

Trong hệ thống của bạn:

* Sharpe giúp so:

  * Mean Reversion vs Trend
  * Strategy A vs B
* **Phải xem cùng**:

  * CAGR
  * Max Drawdown
  * Winrate

📌 Sharpe cao mà DD sâu → nguy hiểm

---

## 🧪 Code tính Sharpe (chuẩn)

```python
def sharpe_ratio(returns, periods_per_year=252):
    mean = returns.mean()
    std = returns.std()
    return (mean / std) * np.sqrt(periods_per_year)
```

---

## 🧩 Sharpe vs các chỉ số khác

| Chỉ số  | Dùng khi      |
| ------- | ------------- |
| Sharpe  | Biến động đều |
| Sortino | Quan tâm lỗ   |
| Calmar  | Trend dài hạn |
| Max DD  | Sống sót      |

---

## 🎯 Kết luận gọn

✔️ **Sharpe = lợi nhuận / rủi ro**
✔️ Dùng để so sánh strategy
⚠️ Không dùng một mình
❌ Sharpe quá cao → nghi ngờ bias

---

