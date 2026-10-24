"""Batch-friendly command line utilities; the Streamlit UI is optional for these commands."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .catalog import DIAGNOSTICS, FORMULAS, maturity_label
from tokamak_analysis.analysis_registry import ANALYSES
from .core import load_table, signal_quality, prepare_time_series, fft_spectrum, welch_psd, sha256_bytes


def main() -> int:
    parser=argparse.ArgumentParser(prog='tis',description='Tokamak Insight Suite command-line tools')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('qc',help='Audit columns and write a JSON report'); p.add_argument('input'); p.add_argument('--output',default='tis_qc.json')
    p=sub.add_parser('fft',help='Compute FFT and Welch PSD from a numeric column'); p.add_argument('input'); p.add_argument('--signal',required=True)
    sampling=p.add_mutually_exclusive_group(required=True); sampling.add_argument('--time'); sampling.add_argument('--fs',type=float)
    p.add_argument('--output',default='tis_fft.csv')
    p=sub.add_parser('catalog',help='Export diagnostic, analysis-maturity, and formula registries'); p.add_argument('--kind',choices=['diagnostics','analyses','formulas','both'],default='both'); p.add_argument('--output-dir',default='.')
    p=sub.add_parser('events',help='Detect ELMs / sawtooth crashes or analyse a current quench'); p.add_argument('input'); p.add_argument('--kind',choices=['elm','sawtooth','quench'],required=True); p.add_argument('--time',required=True); p.add_argument('--signal',required=True); p.add_argument('--n-sigmas',type=float,default=None); p.add_argument('--output',default='tis_events.json')
    p=sub.add_parser('scalings',help='Confinement summary: Greenwald, q95, IPB98(y,2), H98, Martin-08 L-H threshold')
    for flag in ('ip-A','bt-T','r-m','a-m','kappa','ne-line-avg-m3','p-loss-W'): p.add_argument('--'+flag,type=float,required=True)
    p.add_argument('--mass-amu',type=float,default=2.0); p.add_argument('--stored-energy-J',type=float,default=None); p.add_argument('--delta',type=float,default=0.0); p.add_argument('--output',default='tis_scalings.json')
    args=parser.parse_args()
    if args.command=='scalings':
        from .plasma import confinement_summary
        res=confinement_summary(ip_A=args.ip_A,bt_T=args.bt_T,r_m=args.r_m,a_m=args.a_m,kappa=args.kappa,ne_line_avg_m3=args.ne_line_avg_m3,p_loss_W=args.p_loss_W,mass_amu=args.mass_amu,stored_energy_J=args.stored_energy_J,delta=args.delta)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(res,indent=2,ensure_ascii=False,default=float)+'\n',encoding='utf-8')
        print(f'Wrote {output_path.resolve()}'); return 0
    if args.command=='catalog':
        out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
        if args.kind in ('diagnostics','both'):
            diagnostics=[{**row,'maturity_level':maturity_label(row)} for row in DIAGNOSTICS]
            (out/'DIAGNOSTIC_REGISTRY.json').write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        if args.kind == 'analyses':
            analyses=[{**row,'maturity_level':maturity_label(row)} for row in ANALYSES]
            (out/'ANALYSIS_REGISTRY.json').write_text(json.dumps(analyses,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        if args.kind in ('formulas','both'):(out/'FORMULA_REGISTRY.json').write_text(json.dumps(FORMULAS,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(f'Wrote registry file(s) to {out.resolve()}'); return 0
    data=Path(args.input).read_bytes(); frame=load_table(data,Path(args.input).name)
    if args.command=='qc':
        report={'input':str(Path(args.input).name),'sha256':sha256_bytes(data),'rows':len(frame),'columns':list(map(str,frame.columns)),
                'missing_by_column':{str(col): int(count) for col, count in frame.isna().sum().items()},'quality':{}}
        for col in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[col]): report['quality'][str(col)]=signal_quality(frame[col].to_numpy())
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
        print(f"Audited {len(frame)} rows × {len(frame.columns)} columns; report: {Path(args.output).resolve()}"); return 0
    if args.command=='events':
        from .events import detect_elms, detect_sawteeth, current_quench_analysis
        for col in (args.time,args.signal):
            if col not in frame.columns: raise SystemExit(f"Column {col!r} not found. Available: {list(frame.columns)}")
        tt=pd.to_numeric(frame[args.time],errors='coerce').to_numpy(); yy=pd.to_numeric(frame[args.signal],errors='coerce').to_numpy()
        kw={} if args.n_sigmas is None else {'n_sigmas':args.n_sigmas}
        res={'elm':lambda:detect_elms(tt,yy,**kw),'sawtooth':lambda:detect_sawteeth(tt,yy,**kw),'quench':lambda:current_quench_analysis(tt,yy)}[args.kind]()
        def _js(v):
            if isinstance(v,np.ndarray): return v.tolist()
            if isinstance(v,(np.floating,np.integer)): return v.item()
            raise TypeError(type(v))
        report={'input':Path(args.input).name,'sha256':sha256_bytes(data),'kind':args.kind,'result':res}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=_js)+'\n',encoding='utf-8')
        print(f"{args.kind}: wrote {Path(args.output).resolve()}"); return 0
    if args.signal not in frame.columns: raise SystemExit(f"Signal column {args.signal!r} not found. Available: {list(frame.columns)}")
    if args.time and args.time not in frame.columns:
        raise SystemExit(f"Time column {args.time!r} not found. Available: {list(frame.columns)}")
    t=None if not args.time else pd.to_numeric(frame[args.time],errors='coerce').to_numpy()
    tt,yy,fs=prepare_time_series(t,pd.to_numeric(frame[args.signal],errors='coerce').to_numpy(),args.fs)
    f,amp=fft_spectrum(yy,fs); fp,psd=welch_psd(yy,fs)
    # Interpolate PSD on FFT frequencies only for a single export table; preserve both native grids as distinct CSVs.
    base=Path(args.output); base.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'frequency_Hz':f,'amplitude':amp}).to_csv(base,index=False)
    psd_path=base.with_name(base.stem+'_welch_psd.csv'); pd.DataFrame({'frequency_Hz':fp,'psd':psd}).to_csv(psd_path,index=False)
    print(f'Computed FFT and Welch PSD at fs={fs:.8g} Hz; outputs: {base.resolve()} and {psd_path.resolve()}'); return 0

if __name__=='__main__':
    raise SystemExit(main())
