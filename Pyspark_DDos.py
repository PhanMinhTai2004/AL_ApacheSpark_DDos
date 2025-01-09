from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline

# Khởi tạo Spark session
spark = SparkSession.builder \
    .appName("DDoS Attack Detection with KMeans") \
    .getOrCreate()

# Đọc dữ liệu
file_path = "D:/spark_app.py/Data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" 
df = spark.read.csv(file_path, header=True, inferSchema=True)
df = df.select([col(c).alias(c.strip()) for c in df.columns])

# Loại bỏ khoảng trắng thừa trong tên cột
df = df.select([col(c).alias(c.strip()) for c in df.columns])

# Các cột chọn lọc (không bao gồm Label cho phân cụm)
columns_for_features = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets'
]
label_column = 'Label'

# Chuyển đổi các cột thành kiểu float và loại bỏ các giá trị NaN hoặc Infinity
df_selected = df.select([col(c).cast("float") for c in columns_for_features] + [col(label_column)])
df_selected = df_selected.na.drop()

# Chia dữ liệu thành 80% train và 20% test
train_data, test_data = df_selected.randomSplit([0.8, 0.2], seed=42)

# Tạo một vector chứa tất cả các đặc trưng cho phân cụm
vector_assembler = VectorAssembler(inputCols=columns_for_features, outputCol="features")

# Chuẩn hóa dữ liệu (StandardScaler)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# KMeans
kmeans = KMeans(k=2, seed=1, featuresCol="scaled_features", predictionCol="prediction")

# Tạo pipeline
pipeline = Pipeline(stages=[vector_assembler, scaler, kmeans])

# Huấn luyện mô hình trên tập train
model = pipeline.fit(train_data)

# Dự đoán trên tập train và test
result_train = model.transform(train_data)
result_test = model.transform(test_data)

# Gán nhãn DDoS cho các cụm có prediction = 1
result_train = result_train.withColumn(
    "DDoS_Label",
    when(col("prediction") == 1, "DDoS").otherwise("Normal")
)

result_test = result_test.withColumn(
    "DDoS_Label",
    when(col("prediction") == 1, "DDoS").otherwise("Normal")
)

# Hiển thị kết quả phân cụm trên tập train
print("Kết quả phân cụm trên tập train:")
result_train.select('prediction', 'DDoS_Label', label_column, *columns_for_features).show(50)

# Hiển thị kết quả phân cụm trên tập test
print("Kết quả phân cụm trên tập test:")
result_test.select('prediction', 'DDoS_Label', label_column, *columns_for_features).show(50)

spark.stop()