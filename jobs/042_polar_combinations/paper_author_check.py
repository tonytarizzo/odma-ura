"""Optional audit against the Yuan–Duffy–Médard author C kernel. Supply the upstream SOSCL_mex.c locally."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from paper_soft import SoftSCL

SOURCE = "c9d59a0f716b28b5189781d19a19c379228d13101b71e5999fe4c8cb075631ca"
U8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
F64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")


class UNode(ct.Structure):
    _fields_ = [("frz", ct.c_uint8), ("degree", ct.c_uint64), ("index", ct.POINTER(ct.c_uint64))]


def author_kernel(source, folder):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE:
        raise ValueError("Expect original b5500c3 C source (see paper_comparison.md)")
    core = source.read_text().split("/* Mexfunction Interface */", 1)[0]
    core = core.replace("#include <mex.h>", "#include <stdlib.h>\n#include <stdbool.h>")
    (folder/"core.c").write_text(core)
    path = folder/"soscl.so"
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", str(folder/"core.c"), "-o", str(path), "-lm"], check=True)
    dec = ct.CDLL(str(path)).dec
    dec.argtypes = [U8, U8, F64, F64, ct.POINTER(UNode), ct.c_uint64, ct.c_uint64, F64, ct.c_uint8]
    dec.restype = None
    return dec


def check(dec, code, llr):
    n, width = code.n, code.list_size
    u, c = np.zeros((width,n),np.uint8), np.zeros((width,n),np.uint8)
    pm, lost = np.zeros(width), np.zeros(1)
    nodes = (UNode*n)(*[UNode(int(f), 0, None) for f in code.frozen])
    dec(u,c,pm,llr,nodes,n,width,lost,0)
    logz = np.logaddexp(np.logaddexp.reduce(-pm), -lost[0])
    author_share = np.exp(-lost[0]-logz)
    m = np.exp(-pm-logz) @ (1.-2.*c) + author_share*np.tanh(llr/2.)
    local, local_logz, local_share = code.decode_means(llr)
    error = float(np.max(abs(m-local)))
    np.testing.assert_allclose(m,local,rtol=0,atol=1e-9)
    np.testing.assert_allclose((logz,author_share),(local_logz,local_share),rtol=0,atol=1e-9)
    clip = max(np.finfo(float).eps, np.exp(-lost[0]))
    clipped_weights = np.exp(-pm)
    wrapper = (clipped_weights @ (1.-2.*c) + clip*np.tanh(llr/2.))/(clipped_weights.sum()+clip)
    return dict(n=n, width=width, max_mean_error=error, log_z_error=float(abs(logz-local_logz)),
                omitted_share_error=float(abs(author_share-local_share)), logz=logz, omitted_share=author_share,
                wrapper_vs_channel=float(np.max(abs(wrapper-np.tanh(llr/2.)))),
                stable_vs_channel=float(np.max(abs(local-np.tanh(llr/2.)))))


def main(source):
    rows = []
    rng = np.random.default_rng(104208)
    with tempfile.TemporaryDirectory() as td:
        dec = author_kernel(source, Path(td))
        for n,k,crc,L,ns in [(8,3,0,2,4), (16,8,0,4,4), (64,20,4,16,4), (256,68,16,64,3), (512,87,16,64,3)]:
            code = SoftSCL(k,n,crc,L)
            for _ in range(ns):
                data = rng.integers(2,size=k,dtype=np.uint8)
                llr = (1.-2.*code.encode(data))*.8 + rng.normal(0.,1.4,n)
                rows.append(check(dec,code,llr))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    rows = main(args.source)
    args.out.write_text(json.dumps(rows, indent=2)+"\n")
    print("number of synthetic queries", len(rows))
    for n in (8,16,64,256,512):
        group = [r for r in rows if r["n"] == n]
        print(n, "max mean / log Z / omitted-share differences", [max(r[k] for r in group)
              for k in ("max_mean_error","log_z_error","omitted_share_error")])
        print(n, "wrapper and stable maximum change to intrinsic sign", [max(r[k] for r in group)
              for k in ("wrapper_vs_channel","stable_vs_channel")])
