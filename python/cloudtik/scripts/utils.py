import copy
from collections import OrderedDict

import click


def add_command_alias(group, command, name, hidden):
    new_command = copy.deepcopy(command)
    new_command.hidden = hidden
    group.add_command(new_command, name=name)


class NaturalOrderGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        if commands is None:
            commands = OrderedDict()
        elif not isinstance(commands, OrderedDict):
            commands = OrderedDict(commands)
        click.Group.__init__(self, name=name,
                             commands=commands,
                             **attrs)

    def list_commands(self, ctx):
        return self.commands.keys()
