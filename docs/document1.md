Ok, mình sẽ **giải thích `pyproject.toml` theo kiểu “đọc tới đâu hiểu tới đó”**, tập trung vào **những key BẮT BUỘC & hay gặp nhất** cho project **Python / ML / Data / Trading** (đúng ngữ cảnh bạn đang làm).

---

# 🧾 `pyproject.toml` là gì?

* File **trung tâm** của Python project hiện đại
* Quy định:

  * Project tên gì
  * Dùng Python version nào
  * Dependency là gì
  * Package nằm ở đâu
  * Tool (black, isort, pytest, mypy…) cấu hình thế nào

📌 Thay thế:

* `setup.py`
* `setup.cfg`
* 1 phần `requirements.txt`

---

# 1️⃣ `[build-system]` – BẮT BUỘC

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### Ý nghĩa

| Key             | Giải thích                   |
| --------------- | ---------------------------- |
| `requires`      | Tool cần để build project    |
| `build-backend` | Engine dùng để build package |

📌 **99% project dùng nguyên block này**

---

# 2️⃣ `[project]` – TRÁI TIM CỦA FILE

```toml
[project]
name = "project-name"
version = "0.1.0"
description = "ML / Trading system"
readme = "README.md"
requires-python = ">=3.9"
```

### Các key cần hiểu

| Key               | BẮT BUỘC | Ý nghĩa                       |
| ----------------- | -------- | ----------------------------- |
| `name`            | ✅        | Tên package khi `pip install` |
| `version`         | ✅        | Version (semantic)            |
| `description`     | ❌        | Mô tả ngắn                    |
| `readme`          | ❌        | File README                   |
| `requires-python` | ❌        | Version Python cho phép       |

📌 **`name` KHÔNG nhất thiết = tên folder**, nhưng **NÊN GIỐNG** để tránh nhầm import.

---

## 🔹 `dependencies`

```toml
dependencies = [
    "numpy>=1.24",
    "pandas",
    "scikit-learn",
]
```

📌 Tương đương `requirements.txt`

* Tự động cài khi:

```bash
pip install .
```

---

# 3️⃣ `[tool.setuptools]` – CỰC KỲ QUAN TRỌNG (chống lỗi import)

```toml
[tool.setuptools]
package-dir = {"" = "src"}
```

### Ý nghĩa

* Nói với Python:

> "Toàn bộ code nằm trong thư mục `src/`"

📌 Nếu **thiếu block này** → import loạn ngay

---

# 4️⃣ `[tool.setuptools.packages.find]` – Python tìm package ở đâu

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

| Key     | Ý nghĩa                |
| ------- | ---------------------- |
| `where` | Thư mục để tìm package |

📌 Kết hợp với `package-dir` để import ổn định

---

# 5️⃣ `[project.optional-dependencies]` – dependency theo nhóm

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "black",
    "isort",
]

notebook = [
    "jupyter",
    "ipykernel",
]
```

### Cài:

```bash
pip install -e .[dev]
```

📌 Rất hữu ích cho:

* Dev
* Notebook
* Production

---

# 6️⃣ `[project.scripts]` – tạo CLI command

```toml
[project.scripts]
train = "project_name.scripts.train:main"
```

Sau đó:

```bash
train
```

📌 Dùng cho:

* Training pipeline
* Batch job
* Cron

---

# 7️⃣ `[tool.black]`, `[tool.isort]`, `[tool.pytest]`

```toml
[tool.black]
line-length = 88

[tool.isort]
profile = "black"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

📌 Thay cho:

* `.black`
* `.isort.cfg`
* `pytest.ini`

---

# 8️⃣ Những key KHÔNG nên động vào lúc mới

🚫 `[project.entry-points]` (nâng cao)
🚫 `[tool.mypy]` (khi chưa type)
🚫 `[tool.coverage]` (CI/CD)

---

# 9️⃣ File `pyproject.toml` chuẩn cho ML project (MẪU)

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "trading-system"
version = "0.1.0"
description = "Algorithmic trading system"
requires-python = ">=3.9"

dependencies = [
    "numpy",
    "pandas",
    "scikit-learn",
    "mlflow",
]

[project.optional-dependencies]
dev = ["black", "isort", "pytest"]
notebook = ["jupyter", "ipykernel"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

---

# 🔟 Checklist khi đọc `pyproject.toml`

✔ Có `[build-system]`
✔ Có `[project]`
✔ Có `dependencies`
✔ Có `package-dir = "src"`
✔ Có `packages.find where=src`

---

## 👉 Muốn mình làm gì tiếp?

* 🔍 Review `pyproject.toml` của bạn
* 🧱 Tạo template chuẩn cho **Trading / ML / Dashboard**
* 🧪 Giải thích **lỗi import** dựa trên file của bạn

Bạn đang dùng **project nhỏ hay hệ thống lớn (multi-module)**?
