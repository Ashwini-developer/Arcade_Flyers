### **Integrate your Streamlit app with a Kubernetes (K8s) cluster, you'll need to:**

1. Containerize your Streamlit app using the Dockerfile (already done).
2. Push the Docker image to a container registry (like Docker Hub or GCP Container Registry).
3. Create Kubernetes configuration files to deploy your containerized app.
4. Deploy to the Kubernetes cluster using kubectl.

Here’s a detailed step-by-step guide for the Kubernetes integration.

#### **Step 1: Containerize your Streamlit app using the Dockerfile**
##### Build the Docker image
    docker build -t streamlit-nvidia-app .

##### Run the Docker container
    docker run -p 8501:8501 streamlit-nvidia-app
-------------------------------------------------------------------------------------------------------------------------
#### **Step 2: Push the Docker image to a container registry**
If you’re using Docker Hub (replace your-dockerhub-username with your actual Docker Hub username):
    docker tag streamlit-nvidia-app <your-dockerhub-username>/streamlit-nvidia-app:latest

##### Push the image to Docker Hub
    docker push <your-dockerhub-username>/streamlit-nvidia-app:latest

NOTE : If you're using GCP, AWS ECR, or any other registry, follow the respective commands to tag and push the image.
-------------------------------------------------------------------------------------------------------------------------
#### **Step 3: Create Kubernetes YAML Files**
You will need two main files for your Kubernetes deployment:
- **Deployment**: Defines the app (containers, replicas, etc.).
- **Service**: Exposes your app to external users (via LoadBalancer or NodePort).
###### deployment.yaml
This file defines how your containerized Streamlit app will run on the Kubernetes cluster:

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: streamlit-nvidia-app
      labels:
        app: streamlit-nvidia-app
    spec:
      replicas: 2  # Number of pods to run
      selector:
        matchLabels:
          app: streamlit-nvidia-app
      template:
        metadata:
          labels:
            app: streamlit-nvidia-app
        spec:
          containers:
          - name: streamlit-nvidia-app
            image: your-dockerhub-username/streamlit-nvidia-app:latest  # Your pushed Docker image
            ports:
            - containerPort: 8501
            env:
            - name: NVIDIA_API_KEY
              valueFrom:
                secretKeyRef:
                  name: nvidia-api-key-secret
                  key: NVIDIA_API_KEY

###### service.yaml
This file defines how to expose the service, either through a LoadBalancer or NodePort:

    apiVersion: v1
    kind: Service
    metadata:
      name: streamlit-nvidia-app-service
    spec:
      type: LoadBalancer  # Exposes the service externally using cloud provider's load balancer
      selector:
        app: streamlit-nvidia-app
      ports:
        - protocol: TCP
          port: 80  # External port
          targetPort: 8501  # Internal container port

NOTE: If you're running on a local cluster or using Minikube, use NodePort instead of LoadBalancer in the service.yaml file:

    type: NodePort
    
-------------------------------------------------------------------------------------------------------------------------
#### **Step 4: Create Kubernetes Secret for API Key**
You need to store your NVIDIA_API_KEY in a Kubernetes Secret to securely pass the API key to the container:

    kubectl create secret generic nvidia-api-key-secret --from-literal=NVIDIA_API_KEY=your_nvidia_api_key_here

-------------------------------------------------------------------------------------------------------------------------
#### **Step 5: Deploy to the Kubernetes Cluster**
Now that you have the Docker image and Kubernetes manifests ready, you can deploy them to your Kubernetes cluster.

###### Apply the deployment and service configurations:

    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml

###### Check the status of the pods and service:

    kubectl get pods

    kubectl get service streamlit-nvidia-app-service

NOTE : If you're using LoadBalancer, you'll get an external IP once it's assigned.
For NodePort, you'll need the IP of the node and the assigned NodePort to access your application.
-------------------------------------------------------------------------------------------------------------------------
#### **Step 6: Access the Application**
If you used a **LoadBalancer** service type, **Kubernetes** will provision a public IP address. You can check it using:

    kubectl get service streamlit-nvidia-app-service

Once the external IP is available, you can access your Streamlit app by navigating to the external IP on port 80:

    http://<external-ip>

For NodePort, you can access the app using the node IP and NodePort:

    http://<node-ip>:<node-port>

-------------------------------------------------------------------------------------------------------------------------
#### **Step 7: Monitor and Scale the App**

##### To scale up/down the deployment:

    kubectl scale deployment streamlit-nvidia-app --replicas=3

##### To check logs:

    kubectl logs <pod-name>
    

