"""One-frame production-size timing probe, not a decoder performance estimate."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from benchmarks.polar_decoders import build_encoder, normalized_config, fixture, source_hashes
from benchmarks.ura_comparison import noise_variance
from framework.polar_decoders import Scenario, run


def main():
    torch.set_num_threads(2)
    folder=Path(__file__).with_name('results_local_cost')
    folder.mkdir(exist_ok=True)
    output=folder/'timing.json'
    if output.exists():raise FileExistsError(output)
    fingerprint=source_hashes(); rows=[]
    for b in [12,14]:
        c=normalized_config({'profile':'research','B':b,'prefix_bits':b-6,'decoders':['DEC-14'],'eval_frames':1})
        _,_,phi,_=build_encoder(c); ids,noise,signal=fixture(c,phi,26,'iid'); nv=noise_variance(b,4.)
        s=Scenario(phi,signal[0]+np.sqrt(nv)*noise[0],26,nv)
        for i in range(14,25):
            start=time.perf_counter(); counts,meta=run(f'DEC-{i:02}',s); elapsed=time.perf_counter()-start
            truth=np.bincount(ids[0],minlength=2**b)
            row={'B':b,'K':26,'ebn0_db':4,'decoder':f'DEC-{i:02}','seconds':elapsed,
                 'one_frame_pupe':float(1-np.minimum(counts,truth).sum()/26)}
            rows.append(row); print(json.dumps(row),flush=True)
            output.write_text(json.dumps({'scope':'one-frame laptop CPU timing only','source_sha256':fingerprint,'rows':rows},indent=2))


if __name__=='__main__':main()
