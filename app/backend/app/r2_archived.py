import os
import boto3
from botocore.config import Config
from pathlib import Path

R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")

s3_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="auto"
)

def upload(key: str, data: bytes):
    s3_client.put_object(Bucket=R2_BUCKET, Key=key, Body=data)

def download(key: str, dest_path: Path):
    s3_client.download_file(R2_BUCKET, key, str(dest_path))

def presigned_url(key: str, expires: int = 3600) -> str:
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET, "Key": key},
        ExpiresIn=expires
    )

def list_files(prefix: str):
    response = s3_client.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]

def delete_objects(keys: list[str]):
    if not keys:
        return
    # s3_client.delete_objects requires a list of {'Key': '...'}
    delete_list = [{"Key": k} for k in keys]
    s3_client.delete_objects(Bucket=R2_BUCKET, Delete={"Objects": delete_list})

def clear_bucket_except(exclude_keys: list[str] = None):
    if exclude_keys is None:
        exclude_keys = []
        
    # List all objects in the bucket
    paginator = s3_client.get_paginator("list_objects_v2")
    keys_to_delete = []
    for page in paginator.paginate(Bucket=R2_BUCKET):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key not in exclude_keys:
                    keys_to_delete.append(key)
    
    # Delete in batches of 1000 (S3 limit)
    for i in range(0, len(keys_to_delete), 1000):
        delete_objects(keys_to_delete[i:i+1000])
