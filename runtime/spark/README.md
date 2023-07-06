# Compling Spark for CloudTik
CloudTik implement several optimizations to Spark. 

To compiling Spark with these optimizations easily, please follow the below steps.

## 1. Prepare the build environment

The first thing we will do is to git clone the CloudTik repository.
```
git clone https://github.com/oap-project/cloudtik.git && cd cloudtik
```

If the necessary tools needed for building Spark,
follow the Spark building documentation to install.

## 2. Compiling Spark with CloudTik patches

Execute the following command to git clone the Spark repository,
apply the patches and compile:

```
bash ./runtime/spark/scripts/compile-spark.sh --patch
```
