"""Frame IDs preserve the full recording name, including historical suffixes."""
import re

FRAME_ID = re.compile(r'^\d{6}_(?:C\d+M\d+(?:-\d+)?|G\d+)(?:_\d+(?:min|fps))?_\d{8}$')
