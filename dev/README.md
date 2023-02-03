# Developer Guide

## Release procedure
If the version has been bumped up and all source code in main are ready to release,
execute the following procedure to finish a release.

### Step 1: Create the branch and tag for the release version
This step will create the branch and tag based on main. Execute:
```
bash ./dev/release-branch
```
The version information is retrieved from python code "__version__" variable.
Username and the token for github is needed when prompt. 

### Step 2: Build the wheels and release to AWS S3 bucket
This step will build the wheels for different python versions
and upload the wheels to AWS S3 as the backup download location. 
Execute:
```
bash ./dev/release --branch branch-<version>
```

### Step 3: Release docker images (if necessary)
This step will build all the docker images and upload to docker hub
of cloudtik account.
Execute:
```
bash ./dev/release-docker --image-tag <version>
```

### Step 4: Release wheels to PyPI
This step will upload all the wheels in the dist folder under python folder
to PyPI with cloudtik account.
Execute:
```
bash ./dev/release-pip
```
When prompt, input the username and password.
