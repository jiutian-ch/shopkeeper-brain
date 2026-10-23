from dotenv import load_dotenv
load_dotenv(r"D:\Code\shopkeeper_brain\knowledge\.env")
import os
from knowledge.utils.client.storage_clients import StorageClients

BUCKET = "knowledge-base-files"
c = StorageClients.get_minio_client()
print("bucket 存在:", c.bucket_exists(BUCKET))

path = r"D:\Code\shopkeeper_brain_file\tmp_dir\万用表RS-12的使用\auto\images\2219ca75130874ec766983013be86c7afd233a4a1b3a5188990261156f1f2cc5.jpg"
print("本地文件存在:", os.path.exists(path), "大小:", os.path.getsize(path))

with open(path, "rb") as f:
    c.put_object(BUCKET, "_selftest/probe.jpg", f,
                 length=os.path.getsize(path), content_type="image/jpeg")
print("上传成功")

for o in c.list_objects(BUCKET, recursive=True):
    print("桶里:", o.object_name, o.size)