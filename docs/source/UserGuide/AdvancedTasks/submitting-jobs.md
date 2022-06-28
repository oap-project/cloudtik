# Submitting Jobs


### Overview
Cloudtik provides the ability to easily submit tasks to your clusters. Currently, Cloudtik can support submit local script files and web script files, supported file types include ***.sh, .scala, .py, .presto.sql, .trino.sql***.

## Submit Command


```
Usage: cloudtik submit [OPTIONS] /path/to/your-cluster-config.yaml $YOUE_SCRIPT [SCRIPT_ARGS]
```

| COMMAND_OPTIONS| Description|
|---|---|
|--stop|Stop the cluster after the command finishes running.|
|--start|Start the cluster if needed.|
|--screen|Run the command in a screen.|
|--tmux|Run the command in tmux.|
|-n, --cluster-name TEXT|Override the configured cluster name.|
|--no-config-cache|Disable the local cluster config cache.|
|-p, --port-forward INTEGER|ort to forward. Use this multiple times to forward multiple ports.|
|--log-style [auto,record,pretty]|If 'pretty', outputs with formatting and color. If 'record', outputs record-style without formatting. 'auto' defaults to 'pretty', and disables pretty logging if stdin is *not* a TTY.|
|--log-color [auto,false,true]|Use color logging. Auto enables color logging if stdout is a TTY.|


### Specifying arguments for jobs

You can specify additional argument when submitting a job.  
For example, the user has a python file **experiment.py** to submit, and `--smoke-test` is an option for experiment.py. The command is as follows:
```
    cloudtik submit /path/to/your-cluster-config.yaml /path/to/experiment.py  --smoke-test
```
#### Notes 
1. The script file will be automatically synced/downloaded to the path "~/jobs/".
2. Sometimes your parameters for the script will contain special character like ***|,\\***, it will cause the parameters can not be parsed correctly.
 We suggest you use single quote to avoid these problems. For example:
   ''
```    
    --conf spark.oap.sql.columnar.coreRange='"0-31,64-95|32-63,96-127"'  --jars '$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar'
```  


### Submitting background jobs
    
Sometimes, user's network environments may be not stable, cloudtik will be disconnected from the remote clusters during jobs execution.
Or user needs to run some long-time tasks, and just want to check the output halfway or after the job is finished.  
To solve such scenarios, we provide options `--screen` or `--tmux` to support run jobs in background. 
**[Screen](https://www.gnu.org/software/screen/manual/screen.html)** and **[tmux](https://github.com/tmux/tmux/wiki/Getting-Started)** are the most popular Terminal multiplexers, you can choose according to your needs.

####  Using screen
Submitting a job:
```bash
   cloudtik submit --screen /path/to/your-cluster-config.yaml experiment.py
```
Checking background job:
```bash
  # Attaching to your cluster
  cloudtik attach /path/to/your-cluster-config.yaml 
  # Using '-list' to get all screens
  screen -list
  # The output:
  # There is a screen on:
  #      18400..ip-10-10-0-149   (06/27/22 19:11:54)     (Detached)
  # So "18400" is the screen_id
  # Attaching to job screen to view the output of the job.
  screen -r #screen_id
  
```


####  Using tmux
Submitting a job:
```bash
   cloudtik submit --tmux /path/to/your-cluster-config.yaml experiment.py
```
Checking background job:
```bash
  # Attaching to your cluster
  cloudtik attach /path/to/your-cluster-config.yaml 
  # Using 'ls' to get all tmux session
  tmux ls
  # The output:
  # 0: 1 windows (created Mon Jun 27 19:20:02 2022)
  # So "0" is the session-name
  # Attaching to job screen to view the output of the job.
  tmux attach -t #session-name
  
```


