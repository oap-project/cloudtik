# Compling Hadoop for CloudTik
CloudTik requires a few features/fixes which are not provided
by the default Hadoop distribution.
1. Fuse support for mounting HDFS
2. Azure AKS workload identity support in hadoop-azure module

To compiling CloudTik Hadoop easily, please follow the below steps.

## 1. Prepare the build environment
The first thing we will do is to git clone the Apache Hadoop repository
and checkout to the hadoop version branch to build:
```
git clone https://github.com/apache/hadoop.git && cd hadoop
git checkout rel/release-${HADOOP_VERSION}
```
Notice the start-build-env.sh file at the root of the project.
It is a very convenient script that builds and runs a Docker container
in which everything needed for building and testing Hadoop is included.

Execute the following to build the docker image and start it in interactive mode.
```
# the script will mount .m2 and .gnupg, problem if they don't exist
mkdir -p ~/.m2 && touch ~/.gnupg && bash ./start-build-env.sh
```
The above command will build the docker image for hadoop build environment and
run the image with your current user. It suggests you configure your docker to
be able to run without root.

## 2. Compiling Hadoop with CloudTik patches
After the first step is completed. You are in the docker environment which is ready to compile.

The first thing we will do is to git clone the CloudTik repository.
As the current folder is hadoop folder, we move to its parent and clone.
```
cd ..
git clone https://github.com/oap-project/cloudtik.git && cd cloudtik
```
Execute the following command to apply the patches and compile Hadoop:

```
bash ./runtime/hadoop/scripts/compile-hadoop.sh --patch
```
After the build process completed, the tar.gz file is located in the hadoop-dist maven module under the target folder.
The .tar.gz is also available out of the docker container
because the Hadoop source directory was mounted in the docker run command.

## 3. Release
The release-hadoop.sh scripts assumes the current working directory
is the Hadoop repository root. So export CLOUDTIK_HOME pointing to CloudTik
repository root and cd to Hadoop repository root and execute the following,

```
bash $CLOUDTIK_HOME/runtime/hadoop/scripts/release-hadoop.sh
```
