import logging
import os

import boto3
from botocore.exceptions import ClientError


def upload_dataset(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_model(local_dir, bucket_name):
    s3_client = boto3.client("s3")
    model_version = local_dir.split("/")[-2]
    for root, _, files in os.walk(local_dir):
        for filename in files:
            path_to_local_file = os.path.join(root, filename)
            s3_path = f"models/{model_version}/{filename}"
            s3_client.upload_file(path_to_local_file, bucket_name, s3_path)
