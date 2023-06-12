import pkgutil
import importlib

SCRIPT_ALIASES = "_script_aliases_"


class ScriptRegistry:
    """Registry for scripts alias and its target."""
    def __init__(self):
        self._registry = {}

    def register(self, alias, target):
        """Register a command alias provided by this or other packages"""
        self._registry[alias] = target

    def get(self, alias):
        return self._registry.get(alias)


def _register_runtime_aliases():
    from cloudtik import runtime
    # Use pkgutil.walk_packages(runtime.__path__, runtime.__name__ + ".")
    # will also trigger recursively walk of packages
    for loader, module_name, is_pkg in pkgutil.walk_packages(runtime.__path__):
        if is_pkg:
            # This will trigger recursively walk of packages
            # We just want to check the direct module under runtime
            # _module = loader.find_module(module_name).load_module(module_name)
            full_name = runtime.__name__ + '.' + module_name
            _module = importlib.import_module(full_name)
            if SCRIPT_ALIASES in _module.__dict__:
                script_aliases = _module.__dict__[SCRIPT_ALIASES]
                for alias, target in script_aliases.items():
                    _script_registry.register(alias, target)


_script_registry = ScriptRegistry()
_register_runtime_aliases()


def get_registered_script(alias):
    """Get a target script from a registered alias.

    :param alias: The registered alias

    :return: the target of the alias if registered. None if not.
    """
    return _script_registry.get(alias)
