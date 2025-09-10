import os
from typing import Any, List
import datetime
from pytz import timezone
import math
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import custom_job as gca_custom_job_compat
import pprint


### ===== Fill these with your project setup =======  
project = "gpu-launchpad-playground" #project-id
region = "us-central1"  #region
bucket = "salander-us-central1" # Note the bucket must be a SINGLE region bucket type in the same region defined in "region"
reservation_name = "<your reservation name>"  # optional,  fill in if you have any reservation
image_uri="us-central1-docker.pkg.dev/gpu-launchpad-playground/tiangel-customer-workshop/multi-nodes:h100-mega" # Docker image



# GPU setup
n_nodes = 2  # node number. If it's a single node training, set it to 1
machine_type = "a3-megagpu-8g"  
num_gpus_per_node = 8  # GPU number in single node 
gpu_type = "NVIDIA_H100_MEGA_80GB" 

# Job name
timestamp = datetime.datetime.now().astimezone(timezone('US/Pacific')).strftime("%Y%m%d_%H%M%S")
job_name = f"qwen-32B-H100-mega-{timestamp}"

# Training command and args
entrypoint_cmd = ['python', '/trainer/run_train_v2.py','/gcs/salander-us-central1/llama-factory/config/example-qwen-full-sft.yaml']  #replace with your own yaml file address
base_output_dir = os.path.join("/gcs", bucket, job_name)
trainer_args=[]

### ========== Project Setup finished =======  

def launch_job(job_name: str,
               project: str,
               location: str,
               gcs_bucket: str,
               image_uri: str,
               entrypoint_cmd: List[str],
               #trainer_args: List[Any],
               num_nodes: int = 1,
               machine_type: str = "a3-megagpu-8g",
               num_gpus_per_node: int = 8,
               gpu_type: str = "NVIDIA_H100_MEGA_80GB",
               capacity_source: str = "spot"
               #reservation_name: str = ''
               ):
    assert capacity_source in ("on-demand", "spot", "reservation","flex-start")
    aiplatform.init(project=project, location=location, staging_bucket=gcs_bucket)

    train_job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=image_uri,
        command=entrypoint_cmd,
    )

    job_args = dict(
        args=trainer_args,
        replica_count=num_nodes,
        #environment_variables=env_variables,
        machine_type=machine_type,
        accelerator_type=gpu_type,
        accelerator_count=num_gpus_per_node,
        enable_web_access=True,
        boot_disk_size_gb=1000,
        restart_job_on_worker_restart=False,
    )

    if capacity_source == "spot":
        job_args.update(
            {
              "scheduling_strategy": gca_custom_job_compat.Scheduling.Strategy.SPOT,
            }
        )
    elif capacity_source == "reservation":
      assert reservation_name != '', "If using a reservation, provide the reservation_name in the format `projects/{project_id_or_number}/zones/{zone}/reservations/{reservation_name}`"
      job_args.update(
          {
              "reservation_affinity_type": "SPECIFIC_RESERVATION",
              "reservation_affinity_key": "compute.googleapis.com/reservation-name",
              "reservation_affinity_values": [reservation_name],
          }
      )
    elif capacity_source == "flex-start":
       job_args.update(
            {
                # "max_wait_duration": 1800,
                "scheduling_strategy": gca_custom_job_compat.Scheduling.Strategy.FLEX_START,
            }
    )


    pprint.pprint(job_args)
    train_job.submit(**job_args)

launch_job(
    job_name=job_name,
    project=project,
    location=region,
    gcs_bucket=bucket,
    image_uri=image_uri,
    entrypoint_cmd=entrypoint_cmd,
    #trainer_args=trainer_args,
    num_nodes=n_nodes,
    machine_type=machine_type,
    num_gpus_per_node=num_gpus_per_node,
    gpu_type=gpu_type,
    capacity_source="spot", # @param ["spot", "flex-start","reservation"]
    #reservation_name=f"projects/{project}/zones/{zone}/reservations/{reservation_name}",
    )
