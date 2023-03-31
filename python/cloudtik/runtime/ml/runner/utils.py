import logging
import re

logger = logging.getLogger(__name__)

# List of regular expressions to ignore environment variables by.
IGNORE_REGEXES = {'BASH_FUNC_.*', 'OLDPWD', '.*_SECRET_KEY'}


def is_exportable(v):
    return not any(re.match(r, v) for r in IGNORE_REGEXES)
