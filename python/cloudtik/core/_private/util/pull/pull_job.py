import subprocess

from cloudtik.core._private.utils import get_run_script_command


class PullJob:
    def pull(self):
        pass


class ScriptPullJob(PullJob):
    def __init__(self, pull_script, pull_args):
        self.pull_script = pull_script
        self.pull_args = pull_args
        self.pull_cmd = get_run_script_command(
            self.pull_script, self.pull_args)

    def pull(self):
        try:
            subprocess.check_call(self.pull_cmd, shell=True)
        except subprocess.CalledProcessError as err:
            print(f"Called process error {err}")
            raise err
