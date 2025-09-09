This doc provides step-by-step guidance on how to use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to do fine-tuning on Vertex Custom Training, taking an example model of [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct), using [a3-megagpu-8g (H100 mega)](https://cloud.google.com/compute/docs/gpus#h100-gpus) spot capacity type.  Check [this link](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#supported-models) for a full list of supported models on LLaMA-Factory.

## 0. Prerequisite  
1. Familiarity with LLaMA-Factory. Refer to [the documentation](https://llamafactory.readthedocs.io/en/latest/) for comprehensive information. Also, understanding of Distributed Training and model parallelism settings. Consult [this page](https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html#) for additional details.
2. Enable APIs: Verify that the Vertex AI API is enabled for your project.
3. Capacity & Quota: Vertex Custom Training supports Spot/DWS/on-demand/CUD. Make sure you have enough quota for your specific capacity types. 
- On-demand:``aiplatform.googleapis.com/custom_model_training_nvidia_h100_gpus`` 
- Spot/DWS flex-start: ``aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_gpus`` 

## 1. Code & Structure  

To begin, download the necessary code repositories:
``` 
git clone https://github.com/salander0411/vertex-training-llama-factory-recipe.git
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```  

After that the file structure should look like this.    
``` 
Project Root/
├── LLaMA-Factory/ 
├── trainer/
│    └── run_train_v2.py   # actual training script
│    └── util/
├── configurations/        # LLaMA-Factory configuration
│    └── deepspeed/ 
│ 	   └── deepspeed_z3_config.json  
│ ├── llama-70b-full-sft.yaml 
│ └── qwen-full-sft.yaml
├── custom-training-a3-mega.py   # Vertex training script
├── Dockerfile             # Dockerfile
├── Dockerfile.base        # base Dockerfile
└── requirements.txt  
```  

## 2. Upload files to GCS  
This step is optional but recommended, so that you don’t need to package all of the configuration/checkpoints in the Dockerfile, which helps to significantly reduce the Docker file size and reduce the frequency of rebuilding your dockerfile. 
1. Upload the configuration files to GCS. 
2. Upload the model checkpoint to the GCS bucket. 
3. After that,  you could reference your GCS file in Vertex Docker via this format. Please replace all of the `/gcs/` file path with your own.  ``/gcs/salander-us-central1/llama-factory/models/Qwen2.5-VL-32B-Instruct``

## 3. Prepare the Dockerfile  
Build the base Docker image and upload it to your Artifact Registry repository. Replace <Your-region-code> <your-project-id><your-repo-name>  with your own.  The base dockerfile is [referenced from here](https://github.com/hiyouga/LLaMA-Factory/blob/main/docker/docker-cuda/Dockerfile.base). 

```  
docker build -t <your-region-code>-docker.pkg.dev/<your-project-id>/<your-repo-name>/llama-base:v0.1 -f Dockerfile.base .

#example:  
docker build -t us-central1-docker.pkg.dev/gpu-launchpad-playground/tiangel-customer-workshop/llama-base:v0.1 -f Dockerfile.base .

# Artifact Registry repository auth 
gcloud auth configure-docker \
    us-central1-docker.pkg.dev

# upload to Artifact Registry repository 
docker push <your-region-code>-docker.pkg.dev/<your-project-id>/<your-repo-name>/llama-base:v0.1
```  

Revise the Dockerfile #1 line command with your actual base docker image URL
```  
FROM us-central1-docker.pkg.dev/gpu-launchpad-playground/tiangel-customer-workshop/llama-base:v0.1
```  

And build the publish the dockerfile
```
docker build -t <your-region-code>-docker.pkg.dev/<your-project-id>/<your-repo-name>/llama-factory:h100-mega  .

#example:  
docker build -t us-central1-docker.pkg.dev/gpu-launchpad-playground/tiangel-customer-workshop/llama-factory:h100-mega .

# Artifact Registry repository auth 
gcloud auth configure-docker \
    us-central1-docker.pkg.dev

# upload to Artifact Registry repository 
docker push <your-region-code>-docker.pkg.dev/<your-project-id>/<your-repo-name>/llama-factory:h100-mega

```

## 4. Revise and run your training script
In [custom-training-a3-mega.py](https://github.com/salander0411/vertex-training-llama-factory-recipe/blob/main/custom-training-a3-mega.py) , replace everything with your own. Note the sample used spot as a default capacity setting,  if you have a reservation and would like to use it, please replace spot with the reservation type and provide a valid reservation name. 

```   
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

```   

After that run this script to start your Vertex training job   

```
python custom-training-a3-mega.py 
```

## Appendix - Monitoring 

There are several methods that you could track and monitor for your Vertex training job.

### 1.Console
Check the logs & metrics in a custom training job. 

**Metrics**
The metrics including CPU/GPU/Network   
![metrics](images/[vertex%20training]metrics.png)

**Logs**  
Click **view logs** and you will be routed to log explorer, You could write queries to filter out the logs. E.g.,
1. By keyword:  SEARCH("out of memory")
2. By time range: timestamp >= "2023-11-29T23:00:00Z" AND timestamp <= "2023-11-29T23:30:00Z"
3. By severity :  severity = ERROR
4. Regular expressions:  jsonPayload.message =~ "regular expression pattern"

For a full functionality and operator supported  please [check this doc](https://cloud.google.com/logging/docs/view/logging-query-language)

### 2.Interactive Terminal
After enabling the web access in Vertex training script, Vertex will be able to provide a node based debugging terminal for you.  This is very helpful especially if you are used to check the logs in a terminal.  For more information please [check this doc](https://cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell).     

![launch web terminal](images/[vertex%20training]%20launch%20web%20terminal.png)


After entering the terminal.  Note the terminal will be only accessible DURING a training session, and after the training job is complete, you will lose access to this terminal as the node is released automatically.   
![web terminal](images/[vertex%20training]%20web%20terminal.png)

Enabling it by one simple parameter “enable_web_access”.   
![](images/[vertex%20training]%20enable%20web%20access.png)

### 3.Streaming the logs to your bastion  
Make sure gcloud version >= 302.0.0  
```   
# make sure  gcloud version >= 302.0.0
gcloud version
# this command should automatically install necessary libraries , if not please use this doc to install: 
#Command-line interface | Cloud Logging

gcloud alpha logging tail 'resource.labels.job_id="1234567"' \
--format="value(json_payload.message)"

```      
![streaming logs](images/[vertex%20training]%20streaming%20logs.png)   

### 4. 3rd Party Monitoring tool like Tensorboard. 

1. If you use llama-factory, change ‘report_to’ to tensorboard. 
2. Define logging_dir , format as /gcs/salander-europe-west1/xxx/tensorboard  
![tensorboard](images/[vertex%20training]%20enable%20tensorboard.png).  
3. Access the logs in tensorboard via command  similar to ``tensorboard --load_fast=false --logdir=gs://salander-europe-west1/xxx/tensorboard``