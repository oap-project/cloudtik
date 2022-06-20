# Configuring Container Mode
By default, CloudTik is enabled to run with container mode.
And other parameters required for container mode are configured by default.
For advanced users, there may be a need to change these parameters.

## Specifying the docker image to use
By default, CloudTik will use the image tag 'cloudtik/spark-runtime:latest'.
User can specify to use his own image to pull for head and worker nodes.

For example,

```
# Turn on or off container by setting "enabled" to True or False.
docker:
    enabled: True
    image: your-own-image-tag
```

And if you want the head to use a different image,
```
docker:
    enabled: True
    head_image: your-own-head-image-tag
```

And if you want the worker to use a different image,
```
docker:
    enabled: True
    head_image: your-own-worker-image-tag
```

## Specifying the container name
The default docker container name is 'cloudtik-spark'.
You can customize it using the following,

```
docker:
    enabled: True
    container_name: your-container-name
```

## Specifying additional docker run options
You can add additional docker run options using run_options key.

For example,

```
docker:
    enabled: True
    run_options: your-additional-container-run-options
```
Use 'head_run_options' or 'worker_run_options' key if head and worker need to use different run options.

## Specifying initialization commands for container mode
Initialization commands are commands running on the VM host before the container is running.
User can specify initialization commands which will run for only container mode.

For example,

```
docker:
    enabled: True
    initialization_commands:
        - your-initialization-commands-run-for-docker-only-1
        - your-initialization-commands-run-for-docker-only-2
```

If head or worker need to run different initialization commands, use 'head_initialization_commands' or 'worker_initialization_commands' key.
For this case, you can still specify 'initialization_commands' for common commands running both on head and worker.
