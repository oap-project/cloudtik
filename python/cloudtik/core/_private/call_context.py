import copy

from cloudtik.core._private.cli_logger import cli_logger


class CallContext:
    def __init__(self, _cli_logger=cli_logger) -> None:
        """Create a cluster object to operate on with this API.
        """
        self._redirect_output = False  # Whether to log command output to a temporary file
        self._allow_interactive = True  # whether to pass on stdin to running commands.
        self._config = {"use_login_shells": True, "silent_rsync": True}
        self._cli_logger = _cli_logger

    def new_call_context(self):
        new_context = CallContext(_cli_logger=self._cli_logger.new_logger())
        new_context._redirect_output = self._redirect_output
        new_context._allow_interactive = self._allow_interactive
        new_context._config = copy.deepcopy(self._config)
        return new_context

    @property
    def cli_logger(self):
        return self._cli_logger

    def is_output_redirected(self):
        return self._redirect_output

    def set_output_redirected(self, val: bool):
        """Choose between logging to a temporary file and to `sys.stdout`.

        The default is to log to a file.

        Args:
            val (bool): If true, subprocess output will be redirected to
                        a temporary file.
        """
        self._redirect_output = val

    def does_allow_interactive(self):
        return self._allow_interactive

    def set_allow_interactive(self, val: bool):
        """Choose whether to pass on stdin to running commands.

        The default is to pipe stdin and close it immediately.

        Args:
            val (bool): If true, stdin will be passed to command.
        """
        self._allow_interactive = val

    def is_rsync_silent(self):
        return self._config["silent_rsync"]

    def set_rsync_silent(self, val):
        """Choose whether to silence rsync output.

        Most commands will want to list rsync'd files themselves rather than
        print the default rsync spew.
        """
        self._config["silent_rsync"] = val

    def is_using_login_shells(self):
        return self._config["use_login_shells"]

    def set_using_login_shells(self, val):
        """Choose between login and non-interactive shells.

        Non-interactive shells have the benefit of receiving less output from
        subcommands (since progress bars and TTY control codes are not printed).
        Sometimes this can be significant since e.g. `pip install` prints
        hundreds of progress bar lines when downloading.

        Login shells have the benefit of working very close to how a proper bash
        session does, regarding how scripts execute and how the environment is
        set up. This is also how all commands were run in the past. The only reason
        to use login shells over non-interactive shells is if you need some weird
        and non-robust tool to work.

        Args:
            val (bool): If true, login shells will be used to run all commands.
        """
        self._config["use_login_shells"] = val
