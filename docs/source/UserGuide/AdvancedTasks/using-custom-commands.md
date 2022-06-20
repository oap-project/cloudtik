# Using Custom Commands
CloudTik cluster configuration allows user to run his own commands on head or worker at different stages.
This provides a powerful extension for user to do customized installations and configurations.

## Types of commands
CloudTik provides the following types of commands:

- initialization commands
- setup commands
- bootstrap commands
- start commands
- stop commands

## Initialization commands
Initialization commands is useful to do installation or configurations for software needed on the host.
Initialization commands runs on the virtual machine host for both host mode
and container mode.

For example, to specify your own initialization commands,

```
initialization_commands:
    - your-initialization-commands-to-run-1
    - your-initialization-commands-to-run-2
```

The commands specified in 'initialization_commands' will run on the host for
both host mode or container mode. If you want to specify initialization commands
run only for container mode, please refer to [Configuring Container Mode](./configuring-container-mode.md).

If head or worker need to run different initialization commands, use 'head_initialization_commands' or 'worker_initialization_commands' key.
For this case, you can still specify 'initialization_commands' for common commands running both on head and worker.

## Setup commands
Setup commands is used to do installation or configurations for software in the runtime environments,
which means setup commands run on the virtual machine host for host mode, but run within the container
for the container mode.

For example, to specify your own setup commands,

```
setup_commands:
    - your-setup-commands-to-run-1
    - your-setup-commands-to-run-2
```

If head or worker need to run different setup commands, use 'head_setup_commands' or 'worker_setup_commands' key.
For this case, you can still specify 'setup_commands' for common setup commands running both on head and worker.

When user setup commands are run, CloudTik and runtime components are already been installed and configured,
but not yet in running.

## Bootstrap commands
Bootstrap commands run right after setup commands.
They run on VM host for host mode and run within container for container mode.
To make it simple, we don't distinguish the head or worker. It will run on both head and worker nodes.

For example, to specify your own bootstrap commands,

```
bootstrap_commands:
    - your-bootstrap-commands-to-run-1
    - your-bootstrap-commands-to-run-2
```

## Start commands
Start commands run after bootstrap commands for starting runtime services.
They run on VM host for host mode and run within container for container mode.
When user start commands are run, CloudTik services and runtime services have been started and running.

For example, to specify your own start commands,

```
start_commands:
    - your-start-commands-to-run-1
    - your-start-commands-to-run-2
```

If head or worker need to run different start commands, use 'head_start_commands' or 'worker_start_commands' key.
For this case, you can still specify 'start_commands' for common start commands running both on head and worker.

## Stop commands
Stop commands are run for stopping all the services either requested by user or
at the time before the cluster is tearing down.
They run on VM host for host mode and run within container for container mode.


For example, to specify your own stop commands,

```
stop_commands:
    - your-stop-commands-to-run-1
    - your-stop-commands-to-run-2
```

If head or worker need to run different stop commands, use 'head_stop_commands' or 'worker_stop_commands' key.
For this case, you can still specify 'stop_commands' for common stop commands running both on head and worker.
