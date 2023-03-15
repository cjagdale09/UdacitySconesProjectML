#!/usr/bin/env python
# coding: utf-8

# # Deploy and monitor a machine learning workflow for Image Classification

# ## Setting up this notebook
# 
# Notes about the instance size and kernel setup: this notebook has been tested on
# 
# 1. The `Python 3 (Data Science)` kernel
# 2. The `ml.t3.medium` Sagemaker notebook instance
# 
# ## Data Staging
# 
# We'll use a sample dataset called CIFAR to simulate the challenges Scones Unlimited are facing in Image Classification. In order to start working with CIFAR we'll need to:
# 
# 1. Extract the data from a hosting service
# 2. Transform it into a usable shape and format
# 3. Load it into a production system
# 
# In other words, we're going to do some simple ETL!
# 
# ### 1. Extract the data from the hosting service
# 
# In the cell below, define a function `extract_cifar_data` that extracts python version of the CIFAR-100 dataset. The CIFAR dataaset is open source and generously hosted by the University of Toronto at: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz 

# In[1]:


import requests

def extract_cifar_data(url, filename="cifar.tar.gz"):
    """A function for extracting the CIFAR-100 dataset and storing it as a gzipped file
    
    Arguments:
    url      -- the URL where the dataset is hosted
    filename -- the full path where the dataset will be written
    
    """
    
    # Todo: request the data from the data url
    # Hint: use `requests.get` method
    r = requests.get(url)
    with open(filename, "wb") as file_context:
        file_context.write(r.content)
    return


# Let's test it out! Run the following cell and check whether a new file `cifar.tar.gz` is created in the file explorer.

# In[2]:


extract_cifar_data("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")     


# ### 2. Transform the data into a usable shape and format
# 
# Clearly, distributing the data as a gzipped archive makes sense for the hosting service! It saves on bandwidth, storage, and it's a widely-used archive format. In fact, it's so widely used that the Python community ships a utility for working with them, `tarfile`, as part of its Standard Library. Execute the following cell to decompress your extracted dataset:

# In[3]:


import tarfile

with tarfile.open("cifar.tar.gz", "r:gz") as tar:
    tar.extractall()


# A new folder `cifar-100-python` should be created, containing `meta`, `test`, and `train` files. These files are `pickles` and the [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html) provides a simple script that can be used to load them. We've adapted the script below for you to run:

# In[15]:


import pickle

with open("./cifar-100-python/meta", "rb") as f:
    dataset_meta = pickle.load(f, encoding='bytes')

with open("./cifar-100-python/test", "rb") as f:
    dataset_test = pickle.load(f, encoding='bytes')

with open("./cifar-100-python/train", "rb") as f:
    dataset_train = pickle.load(f, encoding='bytes')


# In[16]:


# Feel free to explore the datasets

dataset_train.keys()


# As documented on the homepage, `b'data'` contains rows of 3073 unsigned integers, representing three channels (red, green, and blue) for one 32x32 pixel image per row.

# In[17]:


32*32*3


# For a simple gut-check, let's transform one of our images. Each 1024 items in a row is a channel (red, green, then blue). Each 32 items in the channel are a row in the 32x32 image. Using python, we can stack these channels into a 32x32x3 array, and save it as a PNG file:

# In[18]:


import numpy as np

# Each 1024 in a row is a channel (red, green, then blue)
row = dataset_train[b'data'][0]
red, green, blue = row[0:1024], row[1024:2048], row[2048:]

# Each 32 items in the channel are a row in the 32x32 image
red = red.reshape(32,32)
green = green.reshape(32,32)
blue = blue.reshape(32,32)

# Combine the channels into a 32x32x3 image!
combined = np.dstack((red,green,blue))


# For a more concise version, consider the following:

# In[19]:


# All in one:
test_image = np.dstack((
    row[0:1024].reshape(32,32),
    row[1024:2048].reshape(32,32),
    row[2048:].reshape(32,32)
))


# In[20]:


import matplotlib.pyplot as plt
plt.imshow(test_image);


# Looks like a cow! Let's check the label. `dataset_meta` contains label names in order, and `dataset_train` has a list of labels for each row.

# In[10]:


dataset_train[b'fine_labels'][0]


# Our image has a label of `19`, so let's see what the 19th item is in the list of label names.

# In[11]:


print(dataset_meta[b'fine_label_names'])


# Ok! 'cattle' sounds about right. By the way, using the previous two lines we can do:

# In[12]:


n = 0
print(dataset_meta[b'fine_label_names'][dataset_train[b'fine_labels'][n]])


# Now we know how to check labels, is there a way that we can also check file names? `dataset_train` also contains a `b'filenames'` key. Let's see what we have here:

# In[13]:


print(dataset_train[b'filenames'][0])


# "Taurus" is the name of a subspecies of cattle, so this looks like a pretty reasonable filename. To save an image we can also do:

# In[14]:


plt.imsave("file.png", test_image)


# Your new PNG file should now appear in the file explorer -- go ahead and pop it open to see!
# 
# Now that you know how to reshape the images, save them as files, and capture their filenames and labels, let's just capture all the bicycles and motorcycles and save them. Scones Unlimited can use a model that tells these apart to route delivery drivers automatically.
# 
# In the following cell, identify the label numbers for Bicycles and Motorcycles:

# In[15]:


import pandas as pd

# Todo: Filter the dataset_train and dataset_meta objects to find the label numbers for Bicycle and Motorcycles

for i, name in enumerate(dataset_meta[b'fine_label_names']):
    if name == b'bicycle':
        print(f"Bicycle Row: {i}")
    elif name == b'motorcycle':
        print(f"Motorcycle Row: {i}")


# Good job! We only need objects with label 8 and 48 -- this drastically simplifies our handling of the data! Below we construct a dataframe for you, and you can safely drop the rows that don't contain observations about bicycles and motorcycles. Fill in the missing lines below to drop all other rows:

# In[16]:


#Construct the dataframe
df_train = pd.DataFrame({
    "filenames": dataset_train[b'filenames'],
    "labels": dataset_train[b'fine_labels'],
    "row": range(len(dataset_train[b'filenames']))
})

# Drop all rows from df_train where label is not 8 or 48

df_train = df_train.loc[df_train["labels"].isin([8,48])]

# Decode df_train.filenames so they are regular strings
df_train["filenames"] = df_train["filenames"].apply(
    lambda x: x.decode("utf-8")
)


df_test = pd.DataFrame({
    "filenames": dataset_test[b'filenames'],
    "labels": dataset_test[b'fine_labels'],
    "row": range(len(dataset_test[b'filenames']))
})

# Drop all rows from df_test where label is not 8 or 48

df_test = df_test.loc[df_test["labels"].isin([8,48])]

# Decode df_test.filenames so they are regular strings
df_test["filenames"] = df_test["filenames"].apply(
    lambda x: x.decode("utf-8")
)


# Now that the data is filtered for just our classes, we can save all our images.

# In[17]:


get_ipython().system('mkdir ./train')
get_ipython().system('mkdir ./test')


# In the previous sections we introduced you to several key snippets of code:
# 
# 1. Grabbing the image data:
# 
# ```python
# dataset_train[b'data'][0]
# ```
# 
# 2. A simple idiom for stacking the image data into the right shape
# 
# ```python
# import numpy as np
# np.dstack((
#     row[0:1024].reshape(32,32),
#     row[1024:2048].reshape(32,32),
#     row[2048:].reshape(32,32)
# ))
# ```
# 
# 3. A simple `matplotlib` utility for saving images
# 
# ```python
# plt.imsave(path+row['filenames'], target)
# ```
# 
# Compose these together into a function that saves all the images into the `./test` and `./train` directories. Use the comments in the body of the `save_images` function below to guide your construction of the function:
# 

# In[18]:


def save_images(dataset, path, filename, row):
    try:
        #Grab the image data in row-major form
        img = dataset[b'data'][row]
    except:
        logging.error("Failed to grab image data from dataset.")
        return False
    
    try:
        # Consolidated stacking/reshaping from earlier
        target = np.dstack((
            img[0:1024].reshape(32,32),
            img[1024:2048].reshape(32,32),
            img[2048:].reshape(32,32)
        ))
    except:
        logging.error("Failed to stack and reshape image")
        return False
    
    try:
        # Save the image
        plt.imsave(path+filename, target)
    except:
        logging.error("Failed to save image")
        return False
    
    # Return any signal data you want for debugging
    return True


# In[19]:


## save ALL images
for i,j in zip(df_train['row'].tolist(),df_train['filenames'].tolist()):
    if not save_images(dataset_train, "./train/", j, i):
        logging.error("Func save_images() did not return True")
        break

for i,j in zip(df_test['row'].tolist(),df_test['filenames'].tolist()):
    if not save_images(dataset_test, "./test/", j, i):
        logging.error("Func save_images() did not return True")
        break


# ### 3. Load the data
# 
# Now we can load the data into S3.
# 
# Using the sagemaker SDK grab the current region, execution role, and bucket.

# In[20]:


get_ipython().run_line_magic('pip', 'install --upgrade boto3 sagemaker')


# In[2]:


import sagemaker

session = sagemaker.session.Session()

bucket= session.default_bucket()
print("Default Bucket: {}".format(bucket))

region = session.boto_region_name
print("AWS Region: {}".format(region))

role = sagemaker.get_execution_role()
print("RoleArn: {}".format(role))


# With this data we can easily sync your data up into S3!

# In[22]:


import os

os.environ["DEFAULT_S3_BUCKET"] = bucket
get_ipython().system('aws s3 sync ./train s3://${DEFAULT_S3_BUCKET}/train/')
get_ipython().system('aws s3 sync ./test s3://${DEFAULT_S3_BUCKET}/test/')


# And that's it! You can check the bucket and verify that the items were uploaded.
# 
# ## Model Training
# 
# For Image Classification, Sagemaker [also expects metadata](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html) e.g. in the form of TSV files with labels and filepaths. We can generate these using our Pandas DataFrames from earlier:

# In[23]:


def to_metadata_file(df, prefix):
    df["s3_path"] = df["filenames"]
    df["labels"] = df["labels"].apply(lambda x: 0 if x==8 else 1)
    return df[["row", "labels", "s3_path"]].to_csv(
        f"{prefix}.lst", sep="\t", index=False, header=False
    )
    
to_metadata_file(df_train.copy(), "train")
to_metadata_file(df_test.copy(), "test")


# We can also upload our manifest files:

# In[24]:


import boto3

# Upload files
boto3.Session().resource('s3').Bucket(
    bucket).Object('train.lst').upload_file('./train.lst')
boto3.Session().resource('s3').Bucket(
    bucket).Object('test.lst').upload_file('./test.lst')


# Using the `bucket` and `region` info we can get the latest prebuilt container to run our training job, and define an output location on our s3 bucket for the model. Use the `image_uris` function from the SageMaker SDK to retrieve the latest `image-classification` image below:

# In[25]:


# Use the image_uris function to retrieve the latest 'image-classification' image 
region_name = boto3.Session().region_name
algo_image = sagemaker.image_uris.retrieve(region=region_name, framework="image-classification")
s3_output_location = f"s3://{bucket}/models/image_model"


# We're ready to create an estimator! Create an estimator `img_classifier_model` that uses one instance of `ml.p3.2xlarge`. Ensure that y ou use the output location we defined above - we'll be referring to that later!

# In[26]:


img_classifier_model = sagemaker.estimator.Estimator(
    algo_image,
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    volume_size=30,
    max_run=1*60*60, # terminate if it takes longer than 1 hour
    input_mode="File",
    output_path=s3_output_location,
    sagemaker_session=session,
)


# We can also set a few key hyperparameters and define the inputs for our model:

# In[27]:


img_classifier_model.set_hyperparameters(
    image_shape='3,32,32', 
    num_classes=2, 
    num_training_samples=len(df_train)
)


# The `image-classification` image uses four input channels with very specific input parameters. For convenience, we've provided them below:

# In[28]:


from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput
model_inputs = {
        "train": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/train/",
            content_type="application/x-image"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/test/",
            content_type="application/x-image"
        ),
        "train_lst": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/train.lst",
            content_type="application/x-image"
        ),
        "validation_lst": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/test.lst",
            content_type="application/x-image"
        )
}


# Great, now we can train the model using the model_inputs. In the cell below, call the `fit` method on our model,:

# In[29]:


## TODO: train your model
img_classifier_model.fit(model_inputs)


# If all goes well, you'll end up with a model topping out above `.8` validation accuracy. With only 1000 training samples in the CIFAR dataset, that's pretty good. We could definitely pursue data augmentation & gathering more samples to help us improve further, but for now let's proceed to deploy our model.
# 
# ### Getting ready to deploy
# 
# To begin with, let's configure Model Monitor to track our deployment. We'll define a `DataCaptureConfig` below:

# In[30]:


from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=f"s3://{bucket}/data_capture"
)


# Note the `destination_s3_uri` parameter: At the end of the project, we can explore the `data_capture` directory in S3 to find crucial data about the inputs and outputs Model Monitor has observed on our model endpoint over time.
# 
# With that done, deploy your model on a single `ml.m5.xlarge` instance with the data capture config attached:

# In[31]:


deployment = img_classifier_model.deploy(
    initial_instance_count=1, instance_type='ml.m5.xlarge',
    data_capture_config=data_capture_config
    )

endpoint = deployment.endpoint_name
print(endpoint)


# Note the endpoint name for later as well.
# 
# Next, instantiate a Predictor:

# In[32]:


predictor = deployment


# In the code snippet below we are going to prepare one of your saved images for prediction. Use the predictor to process the `payload`.

# In[33]:


from sagemaker.serializers import IdentitySerializer
import base64

predictor.serializer = IdentitySerializer("image/png")
with open("./test/bicycle_s_001789.png", "rb") as f:
    payload = f.read()

    
inference = predictor.predict(payload)


# Your `inference` object is an array of two values, the predicted probability value for each of your classes (bicycle and motorcycle respectively.) So, for example, a value of `b'[0.91, 0.09]'` indicates the probability of being a bike is 91% and being a motorcycle is 9%.

# In[34]:


print(inference)


# ### Draft Lambdas and Step Function Workflow
# 
# Your operations team uses Step Functions to orchestrate serverless workflows. One of the nice things about Step Functions is that [workflows can call other workflows](https://docs.aws.amazon.com/step-functions/latest/dg/connect-stepfunctions.html), so the team can easily plug your workflow into the broader production architecture for Scones Unlimited.
# 
# In this next stage you're going to write and deploy three Lambda functions, and then use the Step Functions visual editor to chain them together! Our functions are going to work with a simple data object:
# 
# ```python
# {
#     "inferences": [], # Output of predictor.predict
#     "s3_key": "", # Source data S3 key
#     "s3_bucket": "", # Source data S3 bucket
#     "image_data": ""  # base64 encoded string containing the image data
# }
# ```
# 
# A good test object that you can use for Lambda tests and Step Function executions, throughout the next section, might look like this:
# 
# ```python
# {
#   "image_data": "",
#   "s3_bucket": MY_BUCKET_NAME, # Fill in with your bucket
#   "s3_key": "test/bicycle_s_000513.png"
# }
# ```
# 
# Using these fields, your functions can read and write the necessary data to execute your workflow. Let's start with the first function. Your first Lambda function will copy an object from S3, base64 encode it, and then return it to the step function as `image_data` in an event.
# 
# Go to the Lambda dashboard and create a new Lambda function with a descriptive name like "serializeImageData" and select thr 'Python 3.8' runtime. Add the same permissions as the SageMaker role you created earlier. (Reminder: you do this in the Configuration tab under "Permissions"). Once you're ready, use the starter code below to craft your Lambda handler:
# 
# ```python
# import json
# import boto3
# import base64
# 
# s3 = boto3.client('s3')
# 
# def lambda_handler(event, context):
#     """A function to serialize target data from S3"""
#     
#     # Get the s3 address from the Step Function event input
#     key = ## TODO: fill in
#     bucket = ## TODO: fill in
#     
#     # Download the data from s3 to /tmp/image.png
#     ## TODO: fill in
#     
#     # We read the data from a file
#     with open("/tmp/image.png", "rb") as f:
#         image_data = base64.b64encode(f.read())
# 
#     # Pass the data back to the Step Function
#     print("Event:", event.keys())
#     return {
#         'statusCode': 200,
#         'body': {
#             "image_data": image_data,
#             "s3_bucket": bucket,
#             "s3_key": key,
#             "inferences": []
#         }
#     }
# ```
# 
# The next function is responsible for the classification part - we're going to take the image output from the previous function, decode it, and then pass inferences back to the the Step Function.
# 
# Because this Lambda will have runtime dependencies (i.e. the SageMaker SDK) you'll need to package them in your function. *Key reading:* https://docs.aws.amazon.com/lambda/latest/dg/python-package-create.html#python-package-create-with-dependency
# 
# Create a new Lambda function with the same rights and a descriptive name, then fill in the starter code below for your classifier Lambda.
# 
# ```python
# import json
# import sagemaker
# import base64
# from sagemaker.serializers import IdentitySerializer
# 
# # Fill this in with the name of your deployed model
# ENDPOINT = ## TODO: fill in
# 
# def lambda_handler(event, context):
# 
#     # Decode the image data
#     image = base64.b64decode(## TODO: fill in)
# 
#     # Instantiate a Predictor
#     predictor = ## TODO: fill in
# 
#     # For this model the IdentitySerializer needs to be "image/png"
#     predictor.serializer = IdentitySerializer("image/png")
#     
#     # Make a prediction:
#     inferences = ## TODO: fill in
#     
#     # We return the data back to the Step Function    
#     event["inferences"] = inferences.decode('utf-8')
#     return {
#         'statusCode': 200,
#         'body': json.dumps(event)
#     }
# ```
# 
# Finally, we need to filter low-confidence inferences. Define a threshold between 1.00 and 0.000 for your model: what is reasonble for you? If the model predicts at `.70` for it's highest confidence label, do we want to pass that inference along to downstream systems? Make one last Lambda function and tee up the same permissions:
# 
# ```python
# import json
# 
# 
# THRESHOLD = .93
# 
# 
# def lambda_handler(event, context):
#     
#     # Grab the inferences from the event
#     inferences = ## TODO: fill in
#     
#     # Check if any values in our inferences are above THRESHOLD
#     meets_threshold = ## TODO: fill in
#     
#     # If our threshold is met, pass our data back out of the
#     # Step Function, else, end the Step Function with an error
#     if meets_threshold:
#         pass
#     else:
#         raise("THRESHOLD_CONFIDENCE_NOT_MET")
# 
#     return {
#         'statusCode': 200,
#         'body': json.dumps(event)
#     }
# ```
# Once you have tested the lambda functions, save the code for each lambda function in a python script called 'lambda.py'.
# 
# With your lambdas in place, you can use the Step Functions visual editor to construct a workflow that chains them together. In the Step Functions console you'll have the option to author a Standard step function *Visually*.
# 
# When the visual editor opens, you'll have many options to add transitions in your workflow. We're going to keep it simple and have just one: to invoke Lambda functions. Add three of them chained together. For each one, you'll be able to select the Lambda functions you just created in the proper order, filter inputs and outputs, and give them descriptive names.
# 
# Make sure that you:
# 
# 1. Are properly filtering the inputs and outputs of your invokations (e.g. `$.body`)
# 2. Take care to remove the error handling from the last function - it's supposed to "fail loudly" for your operations colleagues!
# 
# Take a screenshot of your working step function in action and export the step function as JSON for your submission package.

# 
# Great! Now you can use the files in `./test` as test files for our workflow. Depending on our threshold, our workflow should reliably pass predictions about images from `./test` on to downstream systems, while erroring out for inferences below our confidence threshold!
# 
# ### Testing and Evaluation
# 
# Do several step function invokations using data from the `./test` folder. This process should give you confidence that the workflow both *succeeds* AND *fails* as expected. In addition, SageMaker Model Monitor will generate recordings of your data and inferences which we can visualize.
# 
# Here's a function that can help you generate test inputs for your invokations:

# In[4]:


import random
import boto3
import json


def generate_test_case():
    # Setup s3 in boto3
    s3 = boto3.resource('s3')
    
    # Randomly pick from sfn or test folders in our bucket
    objects = s3.Bucket(bucket).objects.filter(Prefix = "test")
    
    # Grab any random object key from that folder!
    obj = random.choice([x.key for x in objects])
    
    return json.dumps({
        "image_data": "",
        "s3_bucket": bucket,
        "s3_key": obj
    })
generate_test_case()


# In the Step Function dashboard for your new function, you can create new executions and copy in the generated test cases. Do several executions so that you can generate data you can evaluate and visualize.
# 
# Once you've done several executions, let's visualize the record of our inferences. Pull in the JSONLines data from your inferences like so:

# In[6]:


from sagemaker.s3 import S3Downloader

# In S3 your data will be saved to a datetime-aware path
# Find a path related to a datetime you're interested in
data_path ="s3://sagemaker-us-east-1-990703956453/data_capture/image-classification-2023-03-09-18-44-54-720/AllTraffic/2023/03/09/18/"

S3Downloader.download(data_path, "captured_data")

# Feel free to repeat this multiple times and pull in more data


# The data are in JSONLines format, where multiple valid JSON objects are stacked on top of eachother in a single `jsonl` file. We'll import an open-source library, `jsonlines` that was purpose built for parsing this format.

# In[7]:


get_ipython().system('pip install jsonlines')
import jsonlines


# Now we can extract the data from each of the source files:

# In[8]:


import os

# List the file names we downloaded
file_handles = os.listdir("./captured_data")

# Dump all the data into an array
json_data = []
for jsonl in file_handles:
    with jsonlines.open(f"./captured_data/{jsonl}") as f:
        json_data.append(f.read())


# The data should now be a list of dictionaries, with significant nesting. We'll give you an example of some code that grabs data out of the objects and visualizes it:

# In[9]:


# Define how we'll get our data
def simple_getter(obj):
    inferences = obj["captureData"]["endpointOutput"]["data"]
    timestamp = obj["eventMetadata"]["inferenceTime"]
    return json.loads(inferences), timestamp

simple_getter(json_data[0])


# Finally, here's an example of a visualization you can build with this data. In this last part, you will take some time and build your own - the captured data has the input images, the resulting inferences, and the timestamps.

# In[23]:


# Populate the data for the x and y axis
x = []
y = []
for obj in json_data:
    inference, timestamp = simple_getter(obj)
    
    y.append(max(inference))
    x.append(timestamp)

# Todo: here is an visualization example, take some time to build another visual that helps monitor the result
# Plot the data
plt.scatter(x, y, c=['r' if k<.94 else 'b' for k in y ])
plt.axhline(y=0.94, color='g', linestyle='--')
plt.ylim(bottom=.88)

# Add labels
plt.ylabel("Confidence")
plt.suptitle("Observed Recent Inferences", size=14)
plt.title("Pictured with confidence threshold for production use", size=10)

# Give it some pizzaz!
plt.style.use("Solarize_Light2")
plt.gcf().autofmt_xdate()


# ### Todo: build your own visualization
# 

# In[24]:


def my_simple_getter(obj):
    inferences = obj["captureData"]["endpointOutput"]["data"]
    timestamp = obj["eventMetadata"]["inferenceTime"]
    images = obj["captureData"]["endpointInput"]["data"]
    return json.loads(inferences), timestamp, images

simple_getter(json_data[0])


# In[25]:


import base64
import io
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


# Populate the data for the x and y axis
x = []
y = []
images = []

for obj in json_data:
    inference, timestamp, image = my_simple_getter(obj)
    
    #Filter for images I checked multiple times:
    if image not in images:
        y.append(max(inference))
        x.append(timestamp)
        images.append(image)

# Plot the data
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x, y, c=['r' if k<.94 else 'b' for k in y ])
ax.axhline(y=0.94, color='g', linestyle='--')
plt.ylim(bottom=.50, top=1.1)

# Add labels
plt.ylabel("Confidence", color='black', size=18)
plt.suptitle("Observed Failed Inferences", color='black', size=24)
plt.title("Pictures inset for reference", color='black', size=20)

# Give it some pizzaz!
plt.style.use("Solarize_Light2")
plt.gcf().autofmt_xdate()


from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# Overlay images instead of datapoints

for x0, y0, image in zip(x, y, images):
    if y0 < 0.94:
        # Convert image to usable format:
        image = base64.b64decode(image)
        image = io.BytesIO(image)
        image = mpimg.imread(image, format='PNG')
        
        # Load image in OffsetImage for AnnotationBbox
        im = OffsetImage(image, zoom=2.5)
        
        # Display image at x,y coords:
        ab = AnnotationBbox(im, (x0, y0), frameon=False)
        ax.add_artist(ab)

plt.savefig('jag-visualize.png')


# ### Congratulations!
# 
# You've reached the end of the project. In this project you created an event-drivent ML workflow that can be incorporated into the Scones Unlimited production architecture. You used the SageMaker Estimator API to deploy your SageMaker Model and Endpoint, and you used AWS Lambda and Step Functions to orchestrate your ML workflow. Using SageMaker Model Monitor, you instrumented and observed your Endpoint, and at the end of the project you built a visualization to help stakeholders understand the performance of the Endpoint over time. If you're up for it, you can even go further with these stretch goals:
# 
# * Extend your workflow to incorporate more classes: the CIFAR dataset includes other vehicles that Scones Unlimited can identify with this model.
# * Modify your event driven workflow: can you rewrite your Lambda functions so that the workflow can process multiple image inputs in parallel? Can the Step Function "fan out" to accomodate this new workflow?
# * Consider the test data generator we provided for you. Can we use it to create a "dummy data" generator, to simulate a continuous stream of input data? Or a big paralell load of data?
# * What if we want to get notified every time our step function errors out? Can we use the Step Functions visual editor in conjunction with a service like SNS to accomplish this? Try it out!
# 
# 
# 
