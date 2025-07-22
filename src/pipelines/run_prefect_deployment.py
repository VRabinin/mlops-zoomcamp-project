"""Deploy CRM ingestion flow to Prefect server."""

import logging
from prefect import flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from src.pipelines.run_crm_ingestion import crm_data_ingestion_flow
from src.config.config import get_config


def deploy_crm_ingestion_flow():
    """Deploy CRM data ingestion flow to Prefect server."""
    logger = logging.getLogger(__name__)
    
    # Get configuration
    config = get_config()
    
    try:
        # Create deployment
        deployment = Deployment.build_from_flow(
            flow=crm_data_ingestion_flow,
            name="crm-data-ingestion",
            work_pool_name=config.prefect.work_pool,
            description="CRM sales opportunities data ingestion pipeline",
            tags=["data", "ingestion", "crm", "etl"],
            parameters={},
            # Schedule to run daily at 6 AM
            schedule=CronSchedule(cron="0 6 * * *", timezone="UTC"),
            is_schedule_active=False  # Start with schedule inactive
        )
        
        # Deploy to server
        deployment_id = deployment.apply()
        
        logger.info(f"✅ CRM ingestion flow deployed successfully!")
        logger.info(f"📋 Deployment ID: {deployment_id}")
        logger.info(f"🔧 Work pool: {config.prefect.work_pool}")
        logger.info(f"🚀 To activate schedule, use: prefect deployment set-schedule {deployment_id} --active")
        
        return deployment_id
        
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
    logger.info("🚀 Deploying CRM data ingestion flow to Prefect server")
    
    try:
        deployment_id = deploy_crm_ingestion_flow()
        
        print(f"\n✅ Deployment completed successfully!")
        print(f"📋 Deployment ID: {deployment_id}")
        print(f"\n🎯 Next steps:")
        print(f"1. Start Prefect agent: make prefect-agent")
        print(f"2. Run flow manually: prefect deployment run crm-data-ingestion")
        print(f"3. View in UI: http://localhost:4200")
        
        return 0
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
