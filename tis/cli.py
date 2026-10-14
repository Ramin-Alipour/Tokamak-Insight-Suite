"""Batch-friendly command line utilities; the Streamlit UI is optional for these commands."""
from __future__ import annotations
import argparse
import json
import importlib.util
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .catalog import DIAGNOSTICS, FORMULAS
from .core import load_table, signal_quality, prepare_time_series, fft_spectrum, welch_psd, sha256_bytes


def dashboard_main(streamlit_args: list[str] | None = None) -> int:
    """Launch the installed Streamlit dashboard; optional arguments pass to Streamlit."""
    spec = importlib.util.find_spec("app")
    if spec is None or not spec.origin:
        raise SystemExit("Dashboard module 'app' is not installed. Reinstall tokamak-insight-suite with dashboard dependencies.")
    args = list(sys.argv[1:] if streamlit_args is None else streamlit_args)
    if args and args[0] == "--":
        args = args[1:]
    return subprocess.call([sys.executable, "-m", "streamlit", "run", spec.origin, *args])


def main() -> int:
    parser=argparse.ArgumentParser(prog='tis',description='Tokamak Insight Suite command-line tools')
    sub=parser.add_subparsers(dest='command',required=True)
    dashboard_parser=sub.add_parser('dashboard',help='Launch the Streamlit dashboard')
    dashboard_parser.add_argument('streamlit_args',nargs=argparse.REMAINDER,help='Optional arguments passed to Streamlit after --')
    p=sub.add_parser('qc',help='Audit columns and write a JSON report'); p.add_argument('input'); p.add_argument('--output',default='tis_qc.json')
    p=sub.add_parser('fft',help='Compute FFT and Welch PSD from a numeric column'); p.add_argument('input'); p.add_argument('--signal',required=True); p.add_argument('--time'); p.add_argument('--fs',type=float,default=None); p.add_argument('--output',default='tis_fft.csv')
    p=sub.add_parser('catalog',help='Export diagnostic/formula registry JSON'); p.add_argument('--kind',choices=['diagnostics','formulas','both'],default='both'); p.add_argument('--output-dir',default='.')
    args=parser.parse_args()
    if args.command=='dashboard': return dashboard_main(args.streamlit_args)
    if args.command=='catalog':
        out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
        if args.kind in ('diagnostics','both'):(out/'DIAGNOSTIC_REGISTRY.json').write_text(json.dumps(DIAGNOSTICS,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        if args.kind in ('formulas','both'):(out/'FORMULA_REGISTRY.json').write_text(json.dumps(FORMULAS,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(f'Wrote registry file(s) to {out.resolve()}'); return 0
    data=Path(args.input).read_bytes(); frame=load_table(data,Path(args.input).name)
    if args.command=='qc':
        report={'input':str(Path(args.input).name),'sha256':sha256_bytes(data),'rows':len(frame),'columns':list(map(str,frame.columns)),
                'missing_by_column':frame.isna().sum().to_dict(),'quality':{}}
        for col in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[col]): report['quality'][str(col)]=signal_quality(frame[col].to_numpy())
        Path(args.output).write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
        print(f"Audited {len(frame)} rows × {len(frame.columns)} columns; report: {Path(args.output).resolve()}"); return 0
    if args.signal not in frame.columns: raise SystemExit(f"Signal column {args.signal!r} not found. Available: {list(frame.columns)}")
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
