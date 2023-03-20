# Configuring Custom Environment Variables
Environment variables can be set in many ways.
We don't try to override all these methods.
CloudTik allows you to add custom environment variables when executing every commands
through CloudTik. These include initialization commands, setup commands, bootstrap commands,
start commands, and stop commands. The added environment variables will be set
when running these commands.

## Adding custom environment variables
To adding custom environment variables is simple.
Put your environment name and value as key value map under envs section of runtime.

```
runtime:
    envs:
        YOUR_ENVIRONMENT_VARIABLE_NAME_1: your-environment-variable-value-1
        YOUR_ENVIRONMENT_VARIABLE_NAME_2: your-environment-variable-value-2
```

## Example: Set CloudTik log level to Debug
An example of using environment variable is to set the log level of CloudTik
service processes. The following example set the log level to debug
so that you get more detailed log information when executing "cloudtik monitor" command.

```
runtime:
    envs:
        CLOUDTIK_LOGGING_LEVEL: debug
```
