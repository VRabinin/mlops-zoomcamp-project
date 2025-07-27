"""Deploy CRM ingestion flow to Prefect server using S3 storage with MinIO."""

import logging
import os
import tempfile
import yaml
import boto3
import shutil
from pathlib import Path

from src.config.config import get_config


def deploy_with_s3_storage():
    """Deploy using S3 storage for Prefect 3.x - works with MinIO."""
    logger = logging.getLogger(__name__)
    
    try:
        # Get configuration
        config = get_config()
        
        logger.info("🚀 Creating deployment with S3 storage (MinIO)...")
        
        # Upload source code to S3
        logger.info("📦 Uploading source code to MinIO S3...")
        
        # Create a temporary directory with source code
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy source files to temp directory
            src_dir = Path("src")
            temp_src_dir = Path(temp_dir) / "src"
            shutil.copytree(src_dir, temp_src_dir)
            
            # Also copy config and other dependencies
            config_dir = Path("config")
            if config_dir.exists():
                temp_config_dir = Path(temp_dir) / "config"
                shutil.copytree(config_dir, temp_config_dir)
            
            # Create deployment using S3 storage
            logger.info("🔧 Creating deployment with MinIO S3 storage...")
            
            # Create prefect.yaml with S3 storage using prefect-aws
            yaml_config = {
                'name': 'crm-mlops-project',
                'prefect-version': '3.4.10',
                
                # S3 pull steps using prefect-aws
                'pull': [
                    {
                        'prefect_aws.deployments.steps.pull_from_s3': {
                            'bucket': os.getenv('DATA_LAKE_BUCKET', 'data-lake'),
                            'folder': f"{os.getenv('S3_PREFECT_FLOWS_PATH', 'prefect-flows')}/",
                            'credentials': {
                                'aws_access_key_id': os.getenv('MINIO_ROOT_USER', 'minioadmin'),
                                'aws_secret_access_key': os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin')
                            },
                            'client_parameters': {
                                'endpoint_url': 'http://minio:9000',
                                'region_name': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
                            }
                        }
                    }
                ],
                
                'deployments': [
                    {
                        'name': 'crm-data-ingestion',
                        'entrypoint': 'src/pipelines/run_crm_pipeline.py:crm_data_ingestion_flow',
                        'description': 'CRM sales opportunities data ingestion pipeline (MinIO S3 storage)',
                        'tags': ['data', 'ingestion', 'crm', 'etl', 's3', 'minio'],
                        'parameters': {},
                        'work_pool': {
                            'name': config.prefect.work_pool
                        },
                        'schedule': {
                            'interval': 3600  # Every hour
                        }
                    }
                ]
            }
            
            # Write prefect.yaml
            yaml_file = Path('prefect.yaml')
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False)
            
            logger.info(f"✅ Created MinIO S3 deployment configuration: {yaml_file}")
            
            # Upload source code to MinIO S3 using boto3
            s3_client = boto3.client(
                's3',
                aws_access_key_id='minioadmin',
                aws_secret_access_key='minioadmin',
                endpoint_url='http://localhost:9000',
                region_name='us-east-1'
            )
            
            # Upload all Python files to S3
            prefect_flows_path = os.getenv('S3_PREFECT_FLOWS_PATH', 'prefect-flows')
            data_lake_bucket = os.getenv('DATA_LAKE_BUCKET', 'data-lake')
            
            for py_file in Path(temp_dir).rglob("*.py"):
                relative_path = py_file.relative_to(temp_dir)
                s3_key = f"{prefect_flows_path}/{relative_path}"
                
                logger.info(f"📤 Uploading {relative_path} to MinIO S3...")
                s3_client.upload_file(
                    str(py_file), 
                    data_lake_bucket, 
                    s3_key
                )
            
            # Also upload config files
            for config_file in Path(temp_dir).rglob("*.yaml"):
                relative_path = config_file.relative_to(temp_dir)
                s3_key = f"{prefect_flows_path}/{relative_path}"
                
                logger.info(f"📤 Uploading {relative_path} to MinIO S3...")
                s3_client.upload_file(
                    str(config_file), 
                    data_lake_bucket, 
                    s3_key
                )
            
            logger.info("✅ Source code uploaded to MinIO S3 successfully!")
            
            # Deploy using CLI
            import subprocess
            result = subprocess.run(
                ['.venv/bin/prefect', 'deploy', '--all'],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                env={**os.environ, 'PREFECT_API_URL': 'http://localhost:4200/api'}
            )
            
            if result.returncode == 0:
                logger.info("✅ MinIO S3 deployment successful!")
                logger.info(f"Deploy output: {result.stdout}")
                return "crm-data-ingestion"
            else:
                logger.error(f"❌ MinIO S3 deployment failed: {result.stderr}")
                return None
                
    except Exception as e:
        logger.error(f"❌ MinIO S3 deployment failed: {str(e)}")
        logger.exception("Full error details:")
        return None


def deploy_crm_ingestion_flow():
    """Deploy CRM data ingestion flow using S3 storage with MinIO."""
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting CRM flow deployment with S3 storage...")
    
    # Deploy using S3 storage with file upload
    logger.info("🔄 Deploying with S3 storage and file upload...")
    try:
        result = deploy_with_s3_storage()
        if result:
            logger.info("✅ S3 storage deployment successful!")
            return result
        else:
            raise Exception("S3 storage deployment failed")
    except Exception as e:
        logger.error(f"❌ S3 storage deployment failed: {str(e)}")
        raise Exception("S3 deployment failed")


def main():
    """Main function for deploying the CRM ingestion flow."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Deploying CRM data ingestion flow with S3 storage (MinIO)")
    
    try:
        deployment_name = deploy_crm_ingestion_flow()
        
        print(f"\n✅ S3 deployment completed successfully!")
        print(f"📋 Deployment name: {deployment_name}")
        print(f"\n🎯 Next steps:")
        print(f"1. Check deployment status: prefect deployment ls")
        print(f"2. Run flow manually: prefect deployment run crm_data_ingestion_flow/{deployment_name}")
        print(f"3. View in UI: http://localhost:4200")
        print(f"4. Monitor runs: prefect flow-run ls")
        print(f"5. Check MinIO UI: http://localhost:9001")
        print(f"6. List uploaded files: make minio-list-data")
        print(f"\n💡 Note: Flow code is stored in MinIO S3 and accessible to all workers.")
        
        return 0
        
    except Exception as e:
        print(f"❌ S3 deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Ensure MinIO is running: docker ps | grep minio")
        print(f"2. Check MinIO status: make minio-status")
        print(f"3. Ensure Prefect server is running: make prefect-server")
        print(f"4. Check Prefect worker: docker logs mlops-prefect-setup")
        print(f"5. Verify MinIO buckets: make minio-buckets")
        
        return 1


if __name__ == "__main__":
    exit(main())
