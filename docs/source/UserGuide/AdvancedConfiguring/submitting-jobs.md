# Submitting Jobs to Cluster

## Overview
CloudTik provides the ability to easily submit scripts to run on your clusters. Currently, CloudTik supports to submit local script files and web script files.
The supported file types include:
- .sh: Shell scripts run by bash
- .scala: Scala scripts run by Spark shell
- .py: Python script run by Python
- .presto.sql: Presto SQL scripts run by Presto CLI
- .trino.sql: Trino SQL scripts run by Trino CLI

## Submit Command


```
Usage: cloudtik submit [OPTIONS] /path/to/your-cluster-config.yaml $YOUE_SCRIPT [SCRIPT_ARGS]
```

| Some Key Options        | Description|
|-------------------------|---|
| --screen                |Run the command in a screen.|
| --tmux                  |Run the command in tmux.|
| -n, --cluster-name TEXT |Override the configured cluster name.|
| --no-config-cache       |Disable the local cluster config cache.|


## Specifying additional arguments for the job

You can specify additional argument when submitting a job.  
For example, the user has a python file **experiment.py** to submit, and `--smoke-test` is an option for experiment.py. The command is as follows:
```
    cloudtik submit /path/to/your-cluster-config.yaml /path/to/experiment.py  --smoke-test
```
The script file will be automatically uploaded to the path "~/user/jobs/" on the head. And tehn it will run on head with the interpreter based on the script type.

## Script arguments quote or escaping
If your parameters for the script contain special character like ***|,\\*** or
you need environment variable substitution in the parameters, you need to quote or escape such parameters.
These need to be handled differently as following:

- If the parameter contains special character, you can either quote the parameter by single quote or double quote.
For example,
```
    --conf "value with spaces"
```
or
```
    --conf 'value with spaces'
```

- If the parameter contains local shell environment variable substitution, you need to quote the parameter by double quote.
For example, $HOME will be replaced with the environment variable of $HOME at the local machine.
```
    "$HOME/working"
```

- If the parameter contains remote shell environment variable substitution, you need to quote the parameter by single quote.
For example, $HOME will be replaced with the environment variable of $HOME at the head machine.
```
    '$HOME/runtime'
```

- If the parameter contains '$' character without meaning environment variable substitution, you escape '$' with '\\'.
For this case, you can ues either single quote or double quote.
For example, '\\$abc' will be passed to script as "$abc" without substitution.
```
    '\$abc'
```
or
```
    "\$abc"
```

## Submitting job running in background
    
Sometimes, user's network connection to the cluster may be not stable. CloudTik will be disconnected from the remote clusters during jobs execution.
Or user needs to run some long-time tasks, and just want to check the output halfway or after the job is finished.  
To solve such scenarios, we provide options `--screen` or `--tmux` to support run jobs in background. 
**[Screen](https://www.gnu.org/software/screen/manual/screen.html)** and **[tmux](https://github.com/tmux/tmux/wiki/Getting-Started)** are the most popular Terminal multiplexers, you can choose according to your needs.

###  Using screen
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

###  Using tmux
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


