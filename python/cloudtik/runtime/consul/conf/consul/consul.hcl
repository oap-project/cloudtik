datacenter = "default"
data_dir = "{%data.dir%}"
retry_join = [{%join.list%}]

server = true
bootstrap_expect = {%server.nodes%}
bind_addr = "{%bind.address%}"
client_addr = "{%client.address%}"

ui_config {
  enabled = {%ui.config.enabled%}
}
