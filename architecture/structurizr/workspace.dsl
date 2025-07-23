workspace "CRM Sales Opportunities MLOps Platform" "An end-to-end machine learning platform for CRM sales opportunities prediction" {

    model {
        # External systems
        user = person "Business User" "Sales manager or analyst who needs sales opportunity predictions"
        developer = person "ML Engineer/Developer" "Develops, trains, and deploys ML models"
        dataSource = softwareSystem "CRM System" "External CRM system providing sales and opportunities data" "External System"
        
        # Main MLOps Platform
        mlopsplatform = softwareSystem "MLOps Platform" "End-to-end machine learning platform for sales opportunity prediction" {


            # User Interface Layer
            streamlitApp = container "Streamlit Web App" "User interface for making predictions and viewing insights" "Python, Streamlit" {
                predictionUI = component "Prediction Interface" "Form for inputting sales opportunity data"
                visualizationUI = component "Visualization Dashboard" "Charts and metrics for model performance"
                monitoringUI = component "Monitoring Dashboard" "Real-time model performance monitoring"
            }
            
            # Model Serving Layer
            modelServing = container "Model Serving Service" "Serves trained models for predictions" "Python, MLFlow" {
                modelAPI = component "Model API" "REST API for model inference"
                modelRegistry = component "Model Registry" "Manages model versions and metadata"
                modelLoader = component "Model Loader" "Loads and caches models"
            }
            
            # ML Pipeline Layer
            mlPipeline = container "ML Training Pipeline" "Orchestrates model training and evaluation" "Python, Prefect" {
                dataIngestion = component "Data Ingestion" "Extracts data from CRM system"
                dataPreprocessing = component "Data Preprocessing" "Cleans and transforms data"
                featureEngineering = component "Feature Engineering" "Creates features for ML models"
                modelTraining = component "Model Training" "Trains ML models"
                modelEvaluation = component "Model Evaluation" "Evaluates model performance"
            }
            
            # Orchestration Layer
            orchestrator = container "Workflow Orchestrator" "Manages and schedules ML workflows" "Prefect" {
                scheduler = component "Scheduler" "Schedules training and batch prediction jobs"
                taskExecutor = component "Task Executor" "Executes individual pipeline tasks"
                workflowMonitor = component "Workflow Monitor" "Monitors pipeline execution"
            }
            
            # Experiment Tracking
            experimentTracking = container "Experiment Tracking" "Tracks ML experiments and model versions" "MLFlow" {
                experimentLogger = component "Experiment Logger" "Logs model parameters and metrics"
                artifactStore = component "Artifact Store" "Stores model artifacts and datasets"
                metricsTracker = component "Metrics Tracker" "Tracks model performance metrics"
            }
            
            # Monitoring Layer
            monitoring = container "Model Monitoring" "Monitors model performance and data drift" "Python, Evidently" {
                driftDetector = component "Data Drift Detector" "Detects changes in input data distribution"
                performanceMonitor = component "Performance Monitor" "Monitors model accuracy and performance"
                alertingSystem = component "Alerting System" "Sends alerts for model degradation"
            }
            
            # Data Storage
            dataStorage = container "Data Storage" "Stores training data, features, and model artifacts" "PostgreSQL, S3" {
                featureStore = component "Feature Store" "Stores and serves features"
                dataLake = component "Data Lake" "Raw and processed data storage"
                modelArtifacts = component "Model Artifacts" "Stored model files and metadata"
            }
        }
        
        # Infrastructure
        infrastructure = softwareSystem "Infrastructure" "Cloud and containerization infrastructure" {
            awsCloud = container "AWS Cloud" "Production cloud infrastructure" "AWS" {
                ec2 = component "EC2 Instances" "Compute instances for ML workloads"
                s3 = component "S3 Storage" "Object storage for data and models"
                rds = component "RDS Database" "Managed database service"
                alb = component "Application Load Balancer" "Load balancer for web traffic"
            }
            
            localInfra = container "Local Infrastructure" "Development and testing environment" "Docker, MinIO" {
                dockerServices = component "Docker Services" "Containerized services for local development"
                localstack = component "LocalStack" "Local AWS services emulation"
            }
            
            containerOrchestration = container "Container Orchestration" "Manages containerized applications" "HashiCorp Nomad" {
                nomadCluster = component "Nomad Cluster" "Container orchestration cluster"
                consulService = component "Consul" "Service discovery and configuration"
            }
            
            iacManagement = container "Infrastructure as Code" "Manages infrastructure provisioning" "Terraform" {
                terraformModules = component "Terraform Modules" "Reusable infrastructure components"
                stateManagement = component "State Management" "Terraform state management"
            }
        }
        
        # CI/CD System
        cicdSystem = softwareSystem "CI/CD Pipeline" "Continuous integration and deployment" {
            githubActions = container "GitHub Actions" "CI/CD workflows" "GitHub Actions" {
                buildPipeline = component "Build Pipeline" "Builds and tests application"
                deploymentPipeline = component "Deployment Pipeline" "Deploys to different environments"
                testingPipeline = component "Testing Pipeline" "Runs automated tests"
            }
        }
        
        # Relationships - User interactions
        user -> streamlitApp "Uses web interface to make predictions and view insights"
        developer -> streamlitApp "Monitors model performance and experiments"
        developer -> experimentTracking "Tracks experiments and model versions"
        developer -> orchestrator "Manages ML workflows"
        
        # Data flow relationships
        dataSource -> mlPipeline "Provides CRM and sales data"
        
        # Internal system relationships
        streamlitApp -> modelServing "Requests predictions"
        streamlitApp -> monitoring "Displays monitoring dashboards"
        
        mlPipeline -> experimentTracking "Logs experiments and artifacts"
        mlPipeline -> dataStorage "Stores processed data and features"
        
        orchestrator -> mlPipeline "Orchestrates training workflows"
        orchestrator -> modelServing "Triggers model deployment"
        
        modelServing -> experimentTracking "Retrieves trained models"
        modelServing -> dataStorage "Accesses features for inference"
        
        monitoring -> modelServing "Monitors model predictions"
        monitoring -> dataStorage "Analyzes prediction data"
        
        # Infrastructure relationships
        mlopsplatform -> infrastructure "Deployed on"
        cicdSystem -> infrastructure "Deploys to"
        
        # CI/CD relationships
        githubActions -> mlopsplatform "Builds and deploys"
        githubActions -> infrastructure "Provisions infrastructure"


        local = deploymentEnvironment "Local" {
            deploymentNode "Local Machine" {
                deploymentNode "Docker" {
                    containerInstance streamlitApp
                    containerInstance modelServing
                    containerInstance mlPipeline
                    containerInstance orchestrator
                    containerInstance experimentTracking
                    containerInstance monitoring
                    containerInstance dataStorage
                }
                deploymentNode "LocalStack" {
                    softwareSystemInstance dataSource
                }
            }
        }

        production = deploymentEnvironment "Production" {
            deploymentNode "AWS Cloud" {
                deploymentNode "EC2 Instances" {
                    containerInstance streamlitApp
                    containerInstance modelServing
                    containerInstance mlPipeline
                    containerInstance orchestrator
                    containerInstance experimentTracking
                    containerInstance monitoring
                }
                deploymentNode "RDS" {
                    containerInstance dataStorage
                }
                deploymentNode "S3" {
                    # S3 bucket for data storage
                }
            }
        }
    }

    views {
        systemLandscape "SystemLandscape" {
            include *
            autoLayout
        }

        systemContext mlopsplatform "MLOpsPlatformContext" {
            include *
            autoLayout
        }

        container mlopsplatform "MLOpsPlatformContainers" {
            include *
            autoLayout
        }

        component streamlitApp "StreamlitAppComponents" {
            include *
            autoLayout
        }

        component modelServing "ModelServingComponents" {
            include *
            autoLayout
        }

        component mlPipeline "MLPipelineComponents" {
            include *
            autoLayout
        }

        component monitoring "MonitoringComponents" {
            include *
            autoLayout
        }

#        deployment mlopsplatform "Local" "Local deployment using Docker and LocalStack" {
        deployment mlopsplatform "Local" {
            include *
            autoLayout
        }

#        deployment mlopsplatform "Production" "Production deployment on AWS Cloud" {
        deployment mlopsplatform "Production" {
            include *
            autoLayout
        }

        styles {
            element "Person" {
                color #ffffff
                shape person
                background #08427b
            }
            element "External System" {
                background #999999
                color #ffffff
            }
            element "Software System" {
                background #1168bd
                color #ffffff
            }
            element "Container" {
                background #438dd5
                color #ffffff
            }
            element "Component" {
                background #85bbf0
                color #000000
            }
        }
    }
}
