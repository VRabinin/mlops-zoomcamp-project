"""Deploy CRM ingestion flow to Prefect server using Prefect 3.x."""

import logging
import os
import tempfile
import yaml
from pathlib import Path

from src.config.config import get_config


def create_deployment_yaml():
    """Create a deployment YAML file for Prefect 3.x."""
    logger = logging.getLogger(__name__)
    
    # Get configuration
    config = get_config()
    
    # Create deployment configuration
    deployment_config = {
        'deployments': [
            {
                'name': 'crm-data-ingestion',
                'entrypoint': 'src/pipelines/run_crm_ingestion.py:crm_data_ingestion_flow',
                'description': 'CRM sales opportunities data ingestion pipeline',
                'tags': ['data', 'ingestion', 'crm', 'etl'],
                'parameters': {},
                'work_pool': {
                    'name': config.prefect.work_pool
                },
                'schedule': {
                    'cron': '0 6 * * *',
                    'timezone': 'UTC'
                }
            }
        ]
    }
    
    # Write to deployment file
    deployment_file = Path('prefect.yaml')
    
    with open(deployment_file, 'w') as f:
        yaml.dump(deployment_config, f, default_flow_style=False)
    
    logger.info(f"✅ Created deployment configuration: {deployment_file}")
    return deployment_file


def deploy_with_cli():
    """Deploy using Prefect CLI commands for Prefect 3.x."""
    import subprocess
    import sys
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create the YAML file
        yaml_file = create_deployment_yaml()
        
        # Deploy using prefect CLI
        logger.info("🚀 Deploying using prefect deploy command...")
        
        cmd = [sys.executable, '-m', 'prefect', 'deploy', str(yaml_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            logger.info("✅ Deployment successful!")
            logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"❌ Deployment failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ CLI deployment failed: {str(e)}")
        return False


def deploy_programmatically():
    """Alternative programmatic deployment for Prefect 3.x."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import the flow
        from src.pipelines.run_crm_ingestion import crm_data_ingestion_flow
        
        # Get configuration
        config = get_config()
        
        logger.info("🚀 Attempting programmatic deployment...")
        
        # Try using the flow's deploy method (if available in Prefect 3.x)
        if hasattr(crm_data_ingestion_flow, 'deploy'):
            deployment = crm_data_ingestion_flow.deploy(
                name="crm-data-ingestion",
                work_pool_name=config.prefect.work_pool,
                description="CRM sales opportunities data ingestion pipeline",
                tags=["data", "ingestion", "crm", "etl"],
                cron="0 6 * * *",
                timezone="UTC"
            )
            logger.info(f"✅ Programmatic deployment successful: {deployment}")
            return True
        else:
            logger.warning("Flow.deploy method not available, trying alternative...")
            
            # Alternative: Use flow.serve for local serving
            logger.info("Using flow.serve method for local deployment...")
            crm_data_ingestion_flow.serve(
                name="crm-data-ingestion",
                description="CRM sales opportunities data ingestion pipeline",
                tags=["data", "ingestion", "crm", "etl"],
                cron="0 6 * * *",
                timezone="UTC"
            )
            return True
            
    except Exception as e:
        logger.error(f"❌ Programmatic deployment failed: {str(e)}")
        return False


def deploy_crm_ingestion_flow():
    """Deploy CRM data ingestion flow using the best available method for Prefect 3.x."""
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting CRM flow deployment for Prefect 3.x...")
    
    # Try deployment methods in order of preference
    methods = [
        ("CLI deployment", deploy_with_cli),
        ("Programmatic deployment", deploy_programmatically)
    ]
    
    for method_name, method_func in methods:
        logger.info(f"Trying {method_name}...")
        try:
            if method_func():
                logger.info(f"✅ {method_name} successful!")
                return "crm-data-ingestion"
        except Exception as e:
            logger.warning(f"⚠️ {method_name} failed: {str(e)}")
            continue
    
    raise Exception("All deployment methods failed")

import logging
from datetime import timedelta
from prefect import flow, serve
from prefect.client.schemas.schedules import CronSchedule

from src.pipelines.run_crm_ingestion import crm_data_ingestion_flow
from src.config.config import get_config


def deploy_crm_ingestion_flow():
    """Deploy CRM data ingestion flow to Prefect server using Prefect 3.x serve method."""
    logger = logging.getLogger(__name__)
    
    # Get configuration
    config = get_config()
    
    try:
        # In Prefect 3.x, we use the serve method for deployment
        logger.info("🚀 Deploying CRM ingestion flow using Prefect 3.x serve method...")
        
        # Serve the flow with schedule
        deployment = crm_data_ingestion_flow.serve(
            name="crm-data-ingestion",
            description="CRM sales opportunities data ingestion pipeline",
            tags=["data", "ingestion", "crm", "etl"],
            parameters={},
            # Schedule to run daily at 6 AM (using cron)
            cron="0 6 * * *",
            timezone="UTC",
            # Start with schedule inactive for manual testing
            paused=True
        )
        
        logger.info(f"✅ CRM ingestion flow deployed successfully!")
        logger.info(f"📋 Deployment name: crm-data-ingestion")
        logger.info(f"� Schedule: Daily at 6 AM UTC (currently paused)")
        logger.info(f"🚀 Flow is now served and ready to receive runs")
        
        return "crm-data-ingestion"
        
    except Exception as e:
        logger.error(f"❌ Failed to deploy flow: {str(e)}")
        raise


def main():
    """Main function for deploying the CRM ingestion flow."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Deploying CRM data ingestion flow to Prefect server (3.x)")
    
    try:
        deployment_name = deploy_crm_ingestion_flow()
        
        print(f"\n✅ Deployment completed successfully!")
        print(f"📋 Deployment name: {deployment_name}")
        print(f"\n🎯 Next steps:")
        print(f"1. Check deployment status: prefect deployment ls")
        print(f"2. Run flow manually: prefect deployment run {deployment_name}")
        print(f"3. View in UI: http://localhost:4200")
        print(f"4. Monitor runs: prefect flow-run ls")
        
        return 0
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Ensure Prefect server is running: make prefect-server")
        print(f"2. Check Prefect version: prefect version")
        print(f"3. Try running flow directly: make prefect-run-crm")
        
        return 1


if __name__ == "__main__":
    exit(main())
