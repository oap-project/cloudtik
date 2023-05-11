import logging
import os
import re
import shutil
import io

from itertools import chain

logger = logging.getLogger(__name__)

SUPPORTED_PYTHONS = [(3, 7), (3, 8), (3, 9)]

ROOT_DIR = os.path.dirname(__file__)

PROVIDER_SUBDIR = os.path.join("cloudtik", "providers")
THIRDPARTY_SUBDIR = os.path.join("cloudtik", "thirdparty_files")
TEMPLATES_SUBDIR = os.path.join("cloudtik", "templates")

RUNTIME_SUBDIR = os.path.join("cloudtik", "runtime")

MINIMUM_SUPPORTED_PYTHON_VERSION = "3.8"


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
setup_spec = SetupSpec("cloudtik", "CloudTik: a cloud scale platform for distributed analytics and AI on public clouds")

# NOTE: The lists below must be kept in sync with cloudtik build(.sh)
cloudtik_files = [
    "cloudtik/core/thirdparty/redis/cloudtik-redis-server",
]

# cloudtik default yaml files
cloudtik_files += [
    "cloudtik/core/config-schema.json",
    "cloudtik/core/workspace-schema.json",
    "cloudtik/core/_private/job_waiter/tmux-session.sh",
    "cloudtik/core/_private/job_waiter/screen-session.sh",
]

# If you're adding dependencies for cloudtik extras, please
# also update the matching section of requirements.txt.

setup_spec.extras = {
    "aws": [
        "boto3==1.24.59",
        "s3fs==2022.11.0",
        "botocore",
    ],
    "azure": [
        "azure-cli==2.40.0",
        "azure-identity==1.11.0",
        "azure-storage-blob==12.14.1",
        "azure-storage-file-datalake==12.6.0",
        "azure-mgmt-containerservice",
        "azure-mgmt-privatedns",
        "azure-mgmt-rdbms",
        "adlfs==2023.1.0",
    ],
    "gcp": [
        "google-api-python-client==2.48.0",
        "google-cloud-storage==2.3.0",
        "google-cloud-container==2.21.0",
        "gcsfs==2022.11.0",
    ],
    "aliyun": [
        "alibabacloud_tea_openapi == 0.3.7",
        "alibabacloud_vpc20160428 == 2.0.20",
        "alibabacloud_vpcpeer20220101 == 1.0.6",
        "alibabacloud_ecs20140526 == 3.0.4",
        "alibabacloud_ram20150501 == 1.0.3",
        "alibabacloud_oss20190517 == 1.0.5",
        "ossfs == 2023.1.0",
    ],
    "kubernetes": [
        "kubernetes",
        "urllib3",
        "kopf",
    ],
    "huaweicloud": [
        "huaweicloudsdkecs == 3.1.35",
        "huaweicloudsdkvpc == 3.1.35",
        "huaweicloudsdknat == 3.1.35",
        "huaweicloudsdkeip == 3.1.35",
        "huaweicloudsdkiam == 3.1.35",
        "huaweicloudsdkims == 3.1.35",
        "esdk-obs-python == 3.22.2",
    ],
}

setup_spec.extras["all"] = list(
        set(chain.from_iterable(setup_spec.extras.values())))

setup_spec.extras["eks"] = list(
        set(chain(setup_spec.extras["aws"], setup_spec.extras["kubernetes"])))

setup_spec.extras["aks"] = list(
        set(chain(setup_spec.extras["azure"], setup_spec.extras["kubernetes"])))

setup_spec.extras["gke"] = list(
        set(chain(setup_spec.extras["gcp"], setup_spec.extras["kubernetes"])))

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
    "pycryptodomex",
    "pyopenssl",
    "sshtunnel",
    "colorful",
    "gpustat",
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
    long_description=io.open(
        os.path.join(ROOT_DIR, os.path.pardir, "README.md"), "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/oap-project/cloudtik.git",
    keywords="Distributed Cloud Analytic AI Spark",
    classifiers=[
        f"Programming Language :: Python :: {MINIMUM_SUPPORTED_PYTHON_VERSION}",
    ],
    python_requires=f">={MINIMUM_SUPPORTED_PYTHON_VERSION}",
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
            "cloudtik-operator=cloudtik.providers.kubernetes.cloudtik_operator.operator:main",
            "cloudtik-ganglia=cloudtik.runtime.ganglia.scripts:main",
            "cloudtik-spark=cloudtik.runtime.spark.scripts:main",
            "cloudtik-hdfs=cloudtik.runtime.hdfs.scripts:main",
            "cloudtik-metastore=cloudtik.runtime.metastore.scripts:main",
            "cloudtik-presto=cloudtik.runtime.presto.scripts:main",
            "cloudtik-trino=cloudtik.runtime.trino.scripts:main",
            "cloudtik-zookeeper=cloudtik.runtime.zookeeper.scripts:main",
            "cloudtik-kafka=cloudtik.runtime.kafka.scripts:main",
            "cloudtik-ml=cloudtik.runtime.ml.scripts:main",
            "cloudtik-ml-run=cloudtik.runtime.ml.runner.launch:main",
            "cloudtik-flink=cloudtik.runtime.flink.scripts:main",
            "cloudtik-ray=cloudtik.runtime.ray.scripts:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    license="Apache 2.0") if __name__ == "__main__" else None
