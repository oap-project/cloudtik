import logging
import os
import re
import shutil

from itertools import chain

logger = logging.getLogger(__name__)

SUPPORTED_PYTHONS = [(3, 7), (3, 8), (3, 9)]

ROOT_DIR = os.path.dirname(__file__)

PROVIDER_SUBDIR = os.path.join("cloudtik", "providers")
THIRDPARTY_SUBDIR = os.path.join("cloudtik", "thirdparty_files")
TEMPLATES_SUBDIR = os.path.join("cloudtik", "templates")

RUNTIME_SUBDIR = os.path.join("cloudtik", "runtime")


def find_version(*filepath):
    # Extract version information from filepath
    with open(os.path.join(ROOT_DIR, *filepath)) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


class SetupSpec:
    def __init__(self, name: str, description: str):
        self.name: str = name
        self.version = find_version("cloudtik", "__init__.py")
        self.description: str = description
        self.files_to_include: list = []
        self.install_requires: list = []
        self.extras: dict = {}

    def get_packages(self):
        return setuptools.find_packages()


# "cloudtik" primary wheel package.
setup_spec = SetupSpec("cloudtik", "CloudTik is a cloud scale platform for distributed analytics and AI "
                                   "on public cloud providers including AWS, Azure, GCP, and so on.")

# NOTE: The lists below must be kept in sync with cloudtik build(.sh)
cloudtik_files = [
    "cloudtik/core/thirdparty/redis/redis-server",
]

# cloudtik default yaml files
cloudtik_files += [
    "cloudtik/core/config-schema.json",
    "cloudtik/core/workspace-schema.json",
]

# If you're adding dependencies for cloudtik extras, please
# also update the matching section of requirements.txt.

setup_spec.extras = {
    "aws": [
        "boto3==1.22.13",
        "botocore",
    ],
    "azure": [
        "azure-cli==2.35.0",
        "azure-storage-blob==12.11.0",
        "azure-storage-file-datalake==12.6.0",
    ],
    "gcp": [
        "google-api-python-client==2.48.0",
        "google-cloud-storage==2.3.0",
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

setup_spec.install_requires = [
    "attrs",
    "colorama",
    "click >= 7.0",
    "cryptography>=3.0.0",
    "dataclasses; python_version < '3.7'",
    "filelock",
    "jsonschema",
    "numpy >= 1.16; python_version < '3.9'",
    "numpy >= 1.19.3; python_version >= '3.9'",
    "prometheus_client >= 0.7.1",
    "psutil",
    "pyyaml",
    "redis >= 3.5.0",
    "requests",
    "six",
    "smart_open",
    "prettytable",
    "ipaddr",
    "pycryptodomex"
]


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
        # Preserves file mode (needed to copy executable bit)
        shutil.copy(source, destination, follow_symlinks=True)
        return 1
    return 0


def pip_run(build_ext):
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
    description=setup_spec.description,
    long_description="CloudTik is a cloud scale platform for distributed analytics and AI "
                     "on public cloud providers including AWS, Azure, GCP, and so on. "
                     "CloudTik enables users or enterprises to easily create and manage analytics and AI platform "
                     "on public clouds, with out-of-box optimized functionalities and performance, "
                     "and to go quickly to focus on running the business workloads in minutes or hours.",
    url="https://github.com/oap-project/cloudtik.git",
    keywords="Distributed Cloud Analytic AI Spark",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
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
            "cloudtik-trino=cloudtik.runtime.trino.scripts:main",
            "cloudtik-zookeeper=cloudtik.runtime.zookeeper.scripts:main",
            "cloudtik-kafka=cloudtik.runtime.kafka.scripts:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    license="Apache 2.0") if __name__ == "__main__" else None
