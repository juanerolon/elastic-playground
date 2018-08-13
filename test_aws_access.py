
import boto3
import pandas as pd



if False:
    file= open("/Users/juanerolon/.aws/credentials")
    lst = []
    for line in file:
       lst.append(line)

    print(lst)


if True:
    # Create an S3 client
    s3 = boto3.client('s3')
    # Call S3 to list current buckets
    response = s3.list_buckets()
    # Get a list of all bucket names from the response
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    # Print out the bucket list
    print("Bucket List: %s" % buckets)

if True:

    client = boto3.client('s3') #low-level functional API
    resource = boto3.resource('s3') #high-level object-oriented API
    my_bucket = resource.Bucket('ml-rolon') #subsitute this for your s3 bucket name.
    obj = client.get_object(Bucket='ml-rolon', Key='datasets/cyber/alerts_sample_data.csv')
    df_01 = pd.read_csv(obj['Body'])

print(df_01.columns)
print(df_01.describe())


if False:
    # Create an S3 client
    s3 = boto3.client('s3')

    filename = 'test.txt'
    bucket_name = 'ml-rolon'

    # Uploads the given file using a managed uploader, which will split up large
    # files automatically and upload parts in parallel.
    s3.upload_file(filename, bucket_name, filename)


