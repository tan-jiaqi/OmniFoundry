"""Pipeline stages in execution order. Each stage reads the previous stage's out.jsonl in the same run."""
from . import caption, generate_rov, select, think, verify

ORDER = [select, caption, generate_rov, verify, think]
BY_NAME = {s.NAME: s for s in ORDER}
