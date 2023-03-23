# Integrating OAP with CloudTik

## Creating a CloudTik cluster with OAP 

To install OAP packages by Conda when creating a new CloudTik cluster, please add the following to `bootstrap_commands` section of cluster yaml file.

```buildoutcfg
bootstrap_commands:
    - wget -O ~/bootstrap_oap.sh https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/integrations/oap/bootstrap_oap.sh &&
        bash ~/bootstrap_oap.sh 1.4.0.spark32
```

Then OAP packages will be installed under `$HOME/runtime/oap` on your cluster nodes.
