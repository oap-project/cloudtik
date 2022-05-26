### Configuring GCS Bucket

If you do not already have a GCS bucket, create one by following the 
[guide](https://cloud.google.com/storage/docs/creating-buckets#create_a_new_bucket).

Then login to [Google storage page](https://console.cloud.google.com/storage), once you select your bucket, 
a column will show up in the left of this page, then click **ADD PRINCIPLE** as below.

![gcs-bucket](../../image/gcs-bucket.png)

Fill out the first item with your service account email and select a role: Storage Admin.

![gcs-bucket-principle](../../image/gcs-bucket-principle.png)
