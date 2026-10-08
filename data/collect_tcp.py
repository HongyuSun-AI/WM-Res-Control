import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline_entry import launch
if __name__=='__main__':
    flags=['--stop-at-route-end','--lane-metrics']
    if not any(x.split('=',1)[0]=='--target-speed-kmh' for x in sys.argv[1:]):flags.append('--uncapped-tcp')
    launch('record_tcp_reference',defaults={'--output':'data/collection/tcp/train',
        '--route-id':24795,'--tire-friction':.5,'--weather':'MidRainyNoon','--steps':1200},flags=flags)
