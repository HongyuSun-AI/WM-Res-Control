import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline_entry import launch

if __name__=='__main__':
    defaults={'--output':'data/collection/sac'}
    if not any(x.split('=',1)[0]=='--resume' for x in sys.argv[1:]):
        defaults.update({'--world-model':'demo/models/world.pt','--weather':'MidRainyNoon'})
    launch('train_sac_residual',defaults=defaults)
