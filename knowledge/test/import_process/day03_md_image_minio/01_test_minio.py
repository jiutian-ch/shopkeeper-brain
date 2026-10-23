from minio import Minio

# 初始化客户端
client = Minio(
    "192.168.10.160:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # 是否使用https
)

if not client.bucket_exists("mybucket"):
    client.make_bucket("mybucket")

# 上传文件
client.fput_object(
    bucket_name="mybucket",
    object_name="image/wallhaven-28pjj6.jpg",  # Object名称含路径
    file_path="D:\\Code\\shopkeeper_brain_file\\tmp_dir\\wallhaven-28pjj6.jpg",  # 上传的本地文件路径
    content_type="image/jpeg"  # MIME类型
)

# 上传后访问地址
url = f"http://192.168.10.160:9000/mybucket/image/wallhaven-28pjj6.jpg"
print(url) # 桶的权限要开放  private -> public