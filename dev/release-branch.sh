#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLOUDTIK_HOME=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

CLOUDTIK_BRANCH="main"

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --branch)
        # Override for the branch.
        shift
        CLOUDTIK_BRANCH=$1
        ;;
    *)
        echo "Usage: release-branch.sh [ --branch main ]"
        exit 1
    esac
    shift
done


cd $CLOUDTIK_HOME
git reset --hard
git pull
git checkout ${CLOUDTIK_BRANCH}

CLOUDTIK_VERSION=$(sed -n 's/__version__ = \"\(..*\)\"/\1/p' ./python/cloudtik/__init__.py)

RELEASE_BRANCH="branch-${CLOUDTIK_VERSION}"
RELEASE_TAG="v${CLOUDTIK_VERSION}"

git push origin ${CLOUDTIK_BRANCH}:${RELEASE_BRANCH}
git tag -a ${RELEASE_TAG} -m "CloudTik ${CLOUDTIK_VERSION} Release"
git push origin ${RELEASE_TAG}
