"""Focused comparison panels using the same saved summaries, metrics and bounds."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.polar_decoders import plots

GROUPS={
    'discovery':['REF','D2','ODMA','DEC-02','DEC-04','DEC-05','DEC-14','DEC-15','DEC-16','DEC-17'],
    'importance_refinement':['REF','D2','DEC-05','DEC-17','DEC-18','DEC-19'],
    'particle_refinement':['REF','D2','DEC-07','DEC-20','DEC-21','DEC-22','DEC-23','DEC-24'],
}


def plot_groups(result, out):
    for group,names in GROUPS.items():
        selected=[name for name in names if name in result['config']['decoders']]
        if len(selected)<2:continue
        subset={**result,'config':{**result['config'],'decoders':selected},
                'evaluation':[r for r in result['evaluation'] if r['decoder'] in selected],'training':{}}
        folder=Path(out)/group;folder.mkdir(parents=True,exist_ok=True)
        plots(subset,folder)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    result=json.loads((args.run/'summary.json').read_text())
    plot_groups(result,args.run/'groups')


if __name__=='__main__':main()
