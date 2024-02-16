import boto3
import os
from urllib.parse import urlparse
import subprocess
from dotenv import load_dotenv

def download_and_run_model():
    # Parse CloudCube or S3 URL
    load_dotenv()

    # Set up S3 client
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('CLOUDCUBE_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('CLOUDCUBE_SECRET_ACCESS_KEY'),
                      region_name='us-east-1')

    cloudcube_base_path = 'moygkytojg0o'
    bucket_name='cloud-cube-us2'

    # Download NHLModel.py from S3/CloudCube
    model_key = f"{cloudcube_base_path}/NHLModel.py"

    local_model_path = "NHLModel.py"
    with open(local_model_path, 'wb') as f:
        s3.download_fileobj(bucket_name, model_key, f)

    # Run NHLModel.py script
    subprocess.run(['python', local_model_path], check=True)

if __name__ == '__main__':
    download_and_run_model()
