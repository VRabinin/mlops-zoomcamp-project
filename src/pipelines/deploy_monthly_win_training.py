"""Deploy Monthly Win Probability Training Flow to Prefect.

This script deploys the monthly win probability training flow to Prefect server
with appropriate configuration for different environments.
"""

import logging
from pathlib import Path

from prefect import serve
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from src.config.config import get_config
from src.pipelines.run_monthly_win_training import monthly_win_probability_training_flow


def deploy_monthly_win_training_flow():
    """Deploy the monthly win probability training flow to Prefect."""
    
    # Get configuration
    config = get_config()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Deploying Monthly Win Probability Training Flow")
    
    # Create deployment
    deployment = Deployment.build_from_flow(
        flow=monthly_win_probability_training_flow,
        name="monthly_win_probability_training",
        version="1.0.0",
        description="Train monthly win probability prediction models using MLflow and calibrated classifiers",
        tags=["ml", "training", "monthly-prediction", "crm", "mlops"],
        parameters={
            "snapshot_month": config.first_snapshot_month
        },
        work_pool_name=config.prefect.work_pool,
        # Schedule to run monthly (first day of each month at 2 AM)
        schedule=CronSchedule(cron="0 2 1 * *", timezone="UTC"),
        is_schedule_active=False,  # Start with schedule disabled
    )
    
    # Apply deployment
    deployment_id = deployment.apply()
    
    logger.info(f"✅ Deployment created successfully!")
    logger.info(f"📝 Deployment ID: {deployment_id}")
    logger.info(f"🔧 Work Pool: {config.prefect.work_pool}")
    logger.info(f"📅 Schedule: Monthly (1st day at 2 AM UTC) - Currently DISABLED")
    logger.info(f"🎯 Default Parameters: snapshot_month={config.first_snapshot_month}")
    
    logger.info("\n📋 Next Steps:")
    logger.info("1. Start Prefect agent: make prefect-agent")
    logger.info("2. Run the flow: make prefect-run-monthly-training")
    logger.info("3. Monitor in Prefect UI: http://localhost:4200")
    logger.info("4. Check MLflow for experiments: http://localhost:5005")
    
    return deployment_id


if __name__ == "__main__":
    deploy_monthly_win_training_flow()
