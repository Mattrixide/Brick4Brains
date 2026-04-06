"""Combat robot arena simulator for B4B BattleController testing."""

import os
import sys

# Add parent directory (auto-drive/) to path so state_machine imports work
_auto_drive_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _auto_drive_dir not in sys.path:
    sys.path.insert(0, _auto_drive_dir)
