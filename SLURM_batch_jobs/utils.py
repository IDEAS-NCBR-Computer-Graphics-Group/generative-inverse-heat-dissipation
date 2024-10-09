import re
import os

def is_athena_cluster() -> bool:
    hostname = os.uname().nodename
    pattern = r"^t\d{4}$"
    return bool(re.match(pattern, hostname))
