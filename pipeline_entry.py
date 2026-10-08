import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
IMPLEMENTATION=ROOT/'experiments/low_friction'
def launch(name, defaults=None, flags=()):
    provided={arg.split('=',1)[0] for arg in sys.argv[1:] if arg.startswith('--')}
    for option,value in (defaults or {}).items():
        if option not in provided:sys.argv.extend([option,str(value)])
    for option in flags:
        if option not in provided:sys.argv.append(option)
    sys.path.insert(0,str(IMPLEMENTATION))
    runpy.run_path(str(IMPLEMENTATION/(name+'.py')),run_name='__main__')
