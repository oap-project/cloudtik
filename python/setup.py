import argparse
import errno
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

from enum import Enum
from itertools import chain

logger = logging.getLogger(__name__)

SUPPORTED_PYTHONS = [(3, 6), (3, 7), (3, 8), (3, 9)]

ROOT_DIR = os.path.dirname(__file__)

PROVIDER_SUBDIR = os.path.join("cloudtik", "providers")
THIRDPARTY_SUBDIR = os.path.join("cloudtik", "thirdparty_files")
TEMPLATES_SUBDIR = os.path.join("cloudtik", "templates")

RUNTIME_SUBDIR = os.path.join("cloudtik", "runtime")

exe_suffix = ".exe" if sys.platform == "win32" else ""


def find_version(*filepath):
    # Extract version information from filepath
    with open(os.path.join(ROOT_DIR, *filepath)) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


class SetupType(Enum):
    CLOUDTIK = 1


class BuildType(Enum):
    DEFAULT = 1
    DEBUG = 2
    ASAN = 3
    TSAN = 4


class SetupSpec:
    def __init__(self, type: SetupType, name: str, description: str,
                 build_type: BuildType):
        self.type: SetupType = type
        self.name: str = name
        version = find_version("cloudtik", "__init__.py")
        # add .dbg suffix if debug mode is on.
        if build_type == BuildType.DEBUG:
            self.version: str = f"{version}+dbg"
        elif build_type == BuildType.ASAN:
            self.version: str = f"{version}+asan"
        elif build_type == BuildType.TSAN:
            self.version: str = f"{version}+tsan"
        else:
            self.version = version
        self.description: str = description
        self.build_type: BuildType = build_type
        self.files_to_include: list = []
        self.install_requires: list = []
        self.extras: dict = {}

    def get_packages(self):
        if self.type == SetupType.CLOUDTIK:
            return setuptools.find_packages()
        else:
            return []


build_type = os.getenv("CLOUDTIK_DEBUG_BUILD")
if build_type == "debug":
    BUILD_TYPE = BuildType.DEBUG
elif build_type == "asan":
    BUILD_TYPE = BuildType.ASAN
elif build_type == "tsan":
    BUILD_TYPE = BuildType.TSAN
else:
    BUILD_TYPE = BuildType.DEFAULT

# "cloudtik" primary wheel package.
setup_spec = SetupSpec(
    SetupType.CLOUDTIK, "cloudtik", "CloudTik is a cloud scaling infrastructure for "
                                    "scaling your distributed analytics and AI cluster such as Spark easily on "
                                    "public Cloud environment including AWS, Azure, GCP and so on. ", BUILD_TYPE)

# NOTE: The lists below must be kept in sync with cloudtik build(.sh)
cloudtik_files = [
    "cloudtik/core/thirdparty/redis/redis-server" + exe_suffix,
]

# cloudtik default yaml files
cloudtik_files += [
    "cloudtik/core/config-schema.json",
    "cloudtik/core/workspace-schema.json",
]

# If you're adding dependencies for cloudtik extras, please
# also update the matching section of requirements.txt.

if setup_spec.type == SetupType.CLOUDTIK:
    setup_spec.extras = {
        "aws": [
            "boto3",
            "botocore",
        ],
        "azure": [
            "azure-cli==2.35.0",
            "azure-storage-blob==12.11.0",
            "azure-storage-file-datalake==12.6.0",
        ],
        "gcp": [
            "google-api-python-client",
        ],
        "k8s": [
            "kubernetes",
            "urllib3",
        ],
    }

setup_spec.extras["all"] = list(
        set(chain.from_iterable(setup_spec.extras.values())))

# These are the main dependencies for users of cloudtik. This list
# should be carefully curated. If you change it, please reflect
# the change in the matching section of requirements/requirements.txt
if setup_spec.type == SetupType.CLOUDTIK:
    setup_spec.install_requires = [
        "attrs",
        "colorama",
        "click >= 7.0",
        "cryptography>=3.0.0",
        "dataclasses; python_version < '3.7'",
        "filelock",
        "grpcio >= 1.28.1",
        "jsonschema",
        "msgpack >= 1.0.0, < 2.0.0",
        "numpy >= 1.16; python_version < '3.9'",
        "numpy >= 1.19.3; python_version >= '3.9'",
        "prometheus_client >= 0.7.1",
        "protobuf >= 3.15.3",
        "psutil",
        "pyyaml",
        "redis >= 3.5.0",
        "requests",
        "smart_open",
        "prettytable",
        "ipaddr",
        "pycryptodome"
    ]


def download(url):
    try:
        result = urllib.request.urlopen(url).read()
    except urllib.error.URLError:
        # This fallback is necessary on Python 3.5 on macOS due to TLS 1.2.
        curl_args = ["curl", "-s", "-L", "-f", "-o", "-", url]
        result = subprocess.check_output(curl_args)
    return result


def build(build_python):
    if tuple(sys.version_info[:2]) not in SUPPORTED_PYTHONS:
        msg = ("Detected Python version {}, which is not supported. "
               "Only Python {} are supported.").format(
            ".".join(map(str, sys.version_info[:2])),
            ", ".join(".".join(map(str, v)) for v in SUPPORTED_PYTHONS))
        raise RuntimeError(msg)
    # Note: We are passing in sys.executable so that we use the same
    # version of Python to build packages inside the build.sh script. Note
    # that certain flags will not be passed along such as --user or sudo.
    # TODO: Fix this.
    if not os.getenv("SKIP_THIRDPARTY_INSTALL"):
        pip_packages = ["psutil", "setproctitle==1.2.2", "colorama"]
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install", "-q",
                "--target=" + os.path.join(ROOT_DIR, THIRDPARTY_SUBDIR)
            ] + pip_packages,
            env=dict(os.environ, CC="gcc"))


def walk_directory(directory, exclude_python: bool = False):
    file_list = []
    for (root, dirs, filenames) in os.walk(directory):
        for name in filenames:
            if not exclude_python or not name.endswith(".py"):
                file_list.append(os.path.join(root, name))
    return file_list


def copy_file(target_dir, filename, rootdir):
    # TODO: This feels very brittle. It may not handle all cases. See
    # https://github.com/apache/arrow/blob/master/python/setup.py for an
    # example.
    # File names can be absolute paths, e.g. from walk_directory().
    source = os.path.relpath(filename, rootdir)
    destination = os.path.join(target_dir, source)
    # Create the target directory if it doesn't already exist.
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        if sys.platform == "win32":
            # Does not preserve file mode (needed to avoid read-only bit)
            shutil.copyfile(source, destination, follow_symlinks=True)
        else:
            # Preserves file mode (needed to copy executable bit)
            shutil.copy(source, destination, follow_symlinks=True)
        return 1
    return 0


def add_system_dlls(dlls, target_dir):
    """
    Copy any required dlls required by the c-extension module and not already
    provided by python. They will end up in the wheel next to the c-extension
    module which will guarantee they are available at runtime.
    """
    for dll in dlls:
        # Installing Visual Studio will copy the runtime dlls to system32
        src = os.path.join(r"c:\Windows\system32", dll)
        assert os.path.exists(src)
        shutil.copy(src, target_dir)


def pip_run(build_ext):
    if setup_spec.type == SetupType.CLOUDTIK:
        setup_spec.files_to_include += cloudtik_files
        # Include all non-python files in provider directory
        provider_dir = os.path.join(ROOT_DIR, PROVIDER_SUBDIR)
        setup_spec.files_to_include += walk_directory(provider_dir, True)
        # Include all the thirdparty files
        thirdparty_dir = os.path.join(ROOT_DIR, THIRDPARTY_SUBDIR)
        setup_spec.files_to_include += walk_directory(thirdparty_dir)
        # Include all the configuration template files
        templates_dir = os.path.join(ROOT_DIR, TEMPLATES_SUBDIR)
        setup_spec.files_to_include += walk_directory(templates_dir)
        # Include all the runtime conf and scripts files
        runtime_dir = os.path.join(ROOT_DIR, RUNTIME_SUBDIR)
        setup_spec.files_to_include += walk_directory(runtime_dir, True)

    copied_files = 0
    for filename in setup_spec.files_to_include:
        copied_files += copy_file(build_ext.build_lib, filename, ROOT_DIR)


def api_main(program, *args):
    parser = argparse.ArgumentParser()
    choices = ["build", "python_versions", "clean", "help"]
    parser.add_argument("command", type=str, choices=choices)
    parser.add_argument(
        "-l",
        "--language",
        default="python",
        type=str,
        help="A list of languages to build native libraries. "
             "Supported languages now only include \"python\". "
             "If not specified, only the Python library will be built.")
    parsed_args = parser.parse_args(args)

    result = None

    if parsed_args.command == "build":
        kwargs = dict(build_python=False)
        for lang in parsed_args.language.split(","):
            if "python" in lang:
                kwargs.update(build_python=True)
            else:
                raise ValueError("invalid language: {!r}".format(lang))
        result = build(**kwargs)
    elif parsed_args.command == "python_versions":
        for version in SUPPORTED_PYTHONS:
            # NOTE: On Windows this will print "\r\n" on the command line.
            # Strip it out by piping to tr -d "\r".
            print(".".join(map(str, version)))
    elif parsed_args.command == "clean":
        def onerror(function, path, excinfo):
            nonlocal result
            if excinfo[1].errno != errno.ENOENT:
                msg = excinfo[1].strerror
                logger.error("cannot remove {}: {}".format(path, msg))
                result = 1

        for subdir in THIRDPARTY_SUBDIR:
            shutil.rmtree(os.path.join(ROOT_DIR, subdir), onerror=onerror)
    elif parsed_args.command == "help":
        parser.print_help()
    else:
        raise ValueError("Invalid command: {!r}".format(parsed_args.command))

    return result


if __name__ == "__api__":
    api_main(*sys.argv)

if __name__ == "__main__":
    import setuptools
    import setuptools.command.build_ext


    class BuildExt(setuptools.command.build_ext.build_ext):
        def run(self):
            return pip_run(self)


    class BinaryDistribution(setuptools.Distribution):
        def has_ext_modules(self):
            return True

# Ensure no remaining lib files.
build_dir = os.path.join(ROOT_DIR, "build")
if os.path.isdir(build_dir):
    shutil.rmtree(build_dir)

setuptools.setup(
    name=setup_spec.name,
    version=setup_spec.version,
    author="Intel Corporation",
    author_email="cloudtik-dev@intel.com",
    description=setup_spec.description,
    long_description="CloudTik",
    url="https://github.com/Intel-bigdata/cloudtik",
    keywords="CloudTik package",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=setup_spec.get_packages(),
    cmdclass={"build_ext": BuildExt},
    # The BinaryDistribution argument triggers build_ext.
    distclass=BinaryDistribution,
    install_requires=setup_spec.install_requires,
    setup_requires=["cython >= 0.29.15", "wheel"],
    extras_require=setup_spec.extras,
    entry_points={
        "console_scripts": [
            "cloudtik=cloudtik.scripts.scripts:main",
            "cloudtik-simulator=cloudtik.providers.local.service.cloudtik_cloud_simulator:main",
            "cloudtik-ganglia=cloudtik.runtime.ganglia.scripts:main",
            "cloudtik-spark=cloudtik.runtime.spark.scripts:main",
            "cloudtik-hdfs=cloudtik.runtime.hdfs.scripts:main",
            "cloudtik-metastore=cloudtik.runtime.metastore.scripts:main",
            "cloudtik-presto=cloudtik.runtime.presto.scripts:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    license="Apache 2.0") if __name__ == "__main__" else None
