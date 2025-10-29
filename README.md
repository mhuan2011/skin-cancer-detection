# Skin Cancer Detection

Dự án phân loại ung thư da sử dụng Machine Learning với thuật toán Support Vector Machine (SVM) trên dataset HAM10000.

### 1. Cài đặt thư viện
- Import các thư viện cần thiết như pandas, numpy, matplotlib, seaborn
- Các thư viện machine learning: sklearn, kagglehub
- Thư viện để lưu mô hình: joblib

### 2. Tải và khám phá dữ liệu
- Tải dataset HAM10000 từ Kaggle sử dụng kagglehub
- Khám phá cấu trúc dữ liệu với `data.head()`, `data.info()`
- Kiểm tra dữ liệu bị thiếu với `data.isna().sum()`

### 3. Tiền xử lý dữ liệu (Preprocessing)
- **Xử lý dữ liệu thiếu**: Sử dụng SimpleImputer với strategy="median" để điền giá trị thiếu trong cột tuổi
- **Mapping labels**: Chuyển đổi mã dx thành tên chẩn đoán đầy đủ:
  - `mel` → melanoma
  - `nv` → nevus  
  - `bkl` → benign keratosis-like lesions
  - `bcc` → basal cell carcinoma
  - `akiec` → actinic keratoses
  - `vasc` → vascular lesions
  - `df` → dermatofibroma
- **Encoding**: Sử dụng LabelEncoder để mã hóa các biến categorical (sex, localization)
- **Visualizaion**: Biểu đồ phân phối các loại chẩn đoán và scatter plot theo tuổi, giới tính

### 4. Phân loại nhị phân (Binary Classification)
#### Xử lý mất cân bằng dữ liệu:
- Chia thành 2 nhóm:
  - **Malignant (ác tính)**: ["mel", "bcc", "akiec", "vasc"] → Label 1
  - **Benign (lành tính)**: ["nv", "df", "bkl"] → Label 0
- Sử dụng technique **upsampling** để cân bằng dữ liệu giữa 2 class

#### Xây dựng mô hình:
- **Features**: age, sex_enc, localization_enc
- **Target**: binary_label
- **Train/Test split**: 80/20
- **Scaling**: StandardScaler
- **Model**: SVM với kernel='rbf', C=50, gamma='scale' (đã được tối ưu qua GridSearchCV)

#### Đánh giá mô hình:
- Accuracy score
- Classification report (precision, recall, f1-score)
- Confusion matrix với visualization

### 5. Phân loại đa lớp (Multi-class Classification)
#### Đặc điểm:
- Phân loại tất cả 7 loại ung thư da
- Sử dụng oversampling để cân bằng dữ liệu cho tất cả các class
- **Scaling**: MinMaxScaler
- **Model**: SVM với các tham số tương tự binary classification

### 6. Lưu mô hình
- Lưu mô hình binary classification sử dụng joblib: `svm_binary_skin_cancer.pkl`

## Cấu trúc Project
```
skin-cancer-detection/
├── main.ipynb              
├── README.md             
├── requirements.txt
└── svm_binary_skin_cancer.pkl 

```

## Dataset
- **Nguồn**: HAM10000 từ Kaggle
- **Nội dung**: Metadata của 10,000 hình ảnh da bao gồm thông tin về tuổi, giới tính, vị trí tổn thương và chẩn đoán
- **File chính**: HAM10000_metadata.csv

## Kết quả
- **Binary Classification**: Phân biệt giữa ung thư ác tính và lành tính

