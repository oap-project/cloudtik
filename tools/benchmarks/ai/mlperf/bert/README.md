# Run MLPerf Bert benchmark on Cloudtik cluster

## 1. Create a new Cloudtik cluster with MLPerf toolkit
Before running the Bert preparing data and training. We need to proper tools and
libraries are installed in the cluster created.

Refer to [MLPerf README](../README.md) for detail instructions of this step.

## 2. Prepare data for Bert benchmark
Use "cloudtik status your-cluster-config.yaml" to check the all workers are in ready (update-to-date) status.
If workers are not ready, even you submit a job, the job will still in pending for lack of workers.

Execute the following command to run data preparing script on the cluster:
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'cd $MLPERF_HOME && bash bert/scripts/prepare-data.sh'
```
Replace the cluster configuration file in the above command for your case.

The above command will submit and run the job in foreground and possible need a long time.
you may need to run the command with --tmux option for background execution
for avoiding terminal disconnection in the middle. And you don't get the command result.
Please refer to [CloudTik Submitting Jobs](https://cloudtik.readthedocs.io/en/latest/UserGuide/AdvancedConfigurations/submitting-jobs.html) for
the details for run job in background.

## 3. Run MLPerf Bert benchmark
Execute the following command to run Bert training benchmark 
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'cd $MLPERF_HOME && bash bert/scripts/train.sh'
```
