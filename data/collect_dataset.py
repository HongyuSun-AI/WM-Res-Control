import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline_entry import launch
if __name__=='__main__':
    launch('setup_environment',defaults={'--record':'data/collection/world/train',
        '--seed':42,'--steps':6000,'--weather':'MidRainyNoon'},flags=['--navigate'])
