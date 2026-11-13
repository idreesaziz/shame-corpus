"""Shared, account-independent locations for the public reproduction workflow."""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get('SHAME_WORKSPACE', str(REPOSITORY / '.artifacts' / 'workspace'))).resolve()
