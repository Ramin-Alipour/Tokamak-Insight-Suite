#!/usr/bin/env python3
"""Discover GOLEM shot 30622 probe signals from Data.php and known data APIs."""
from __future__ import annotations
import hashlib, json, re, io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np

SHOT = 30622
ROOT = "http://golem.fjfi.cvut.cz"
OUT = Path("artifacts/golem-probe-30622")
OUT.mkdir(parents=True, exist_ok=True)

class TableParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows=[]; self.row=None; self.cell=None; self.href=None; self.links=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag.lower()=="tr": self.row=[]
        elif tag.lower() in ("td","th") and self.row is not None: self.cell=[]
        elif tag.lower()=="a" and a.get("href"): self.href={"href":a["href"],"text":""}
    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)
        if self.href is not None: self.href["text"] += data
    def handle_endtag(self, tag):
        if tag.lower() in ("td","th") and self.row is not None and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split())); self.cell=None
        elif tag.lower()=="tr" and self.row is not None:
            if self.row: self.rows.append(self.row)
            self.row=None
        elif tag.lower()=="a" and self.href is not None:
            self.links.append(self.href); self.href=None

def get(url, timeout=5):
    req=Request(url,headers={"User-Agent":"Tokamak-Insight-Suite-probe-discovery/1.1"})
    with urlopen(req,timeout=timeout) as r: return r.read(),r.geturl(),r.headers.get("Content-Type","")

report={"shot":SHOT,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"Find direct swept BPP voltage/current or determine whether only unswept probe channels are published.",
        "data_page":f"{ROOT}/shots/{SHOT}/Data.php","page_attempts":[],"matching_table_rows":[],"candidate_results":[]}

pages=[f"{ROOT}/shots/{SHOT}/Data.php",f"{ROOT}/shots/{SHOT}/",f"https://golem.fjfi.cvut.cz/shots/{SHOT}/Data.php",
       f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/",
       f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/data/",
       f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/analysis.html"]
identifiers=set(["bpp","lp","BPP","LP","U_bias","U_current","bpp_voltage","bpp_current",
    "probe_voltage","probe_current","PetiProbe:U_bias","PetiProbe:U_current","PetiProbe/U_bias",
    "PetiProbe/U_current","nistandard","electron_temperature","toroidal_field","swept_bpp",
    "bpp_current","bpp_voltage","ballpen_voltage","ballpen_current"])
for page in pages:
    rec={"url":page}
    try:
        raw,final,ctype=get(page)
        rec.update(status="downloaded",final_url=final,content_type=ctype,bytes=len(raw),
                   sha256=hashlib.sha256(raw).hexdigest())
        parser=TableParser(); parser.feed(raw.decode("utf-8","replace"))
        for row in parser.rows:
            joined=" | ".join(row)
            if re.search(r"bpp|ball.?pen|probe|nistandard|u_bias|u_current|sweep|langmuir",joined,re.I):
                report["matching_table_rows"].append({"page":final,"cells":row})
                for cell in row:
                    cell=cell.strip()
                    if re.fullmatch(r"[A-Za-z0-9_:/.-]{2,100}",cell) and re.search(r"bpp|ball.?pen|probe|nistandard|u_bias|u_current|sweep|langmuir",cell,re.I):
                        identifiers.add(cell)
        for link in parser.links:
            href=link["href"]
            joined=href+" "+link["text"]
            is_probe_link=bool(re.search(r"bpp|ball.?pen|probe|nistandard|u_bias|u_current|sweep|langmuir",joined,re.I))
            is_das_link="/DAS/" in href
            is_das_page=("DAS/0512NIstandard.ONN" in page)
            if is_probe_link or is_das_page or is_das_link:
                report.setdefault("matching_links",[]).append({"page":final,**link})
            if is_probe_link or is_das_page or is_das_link:
                tail=href.rstrip("/").split("/")[-1]
                if re.fullmatch(r"[A-Za-z0-9_:/.-]{2,100}",tail): identifiers.add(tail)
                if href.startswith(("http://","https://")): report.setdefault("discovered_urls",[]).append(href)
                elif href.startswith("/"): report.setdefault("discovered_urls",[]).append(ROOT+href)
                else: report.setdefault("discovered_urls",[]).append(final.rsplit("/",1)[0]+"/"+href)
        (OUT/f"page_{len(report['page_attempts']):02d}.html").write_bytes(raw)
    except Exception as exc:
        rec.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    report["page_attempts"].append(rec)

# Follow every DAQ directory linked from the shot page. This checks whether the
# swept BPP signal is in a high-speed scope rather than the low-rate NIstandard channel.
das_pages = list(dict.fromkeys(
    x["href"] if x["href"].startswith(("http://","https://")) else ROOT+x["href"]
    for x in report.get("matching_links", []) if "/DAS/" in x["href"]
))
report["daq_directory_attempts"] = []
for das_page in das_pages:
    try:
        raw, final, ctype = get(das_page)
        parser = TableParser()
        parser.feed(raw.decode("utf-8","replace"))
        report["daq_directory_attempts"].append({"url":das_page,"status":"downloaded","bytes":len(raw)})
        (OUT / ("daq_" + hashlib.sha256(das_page.encode()).hexdigest()[:8] + ".html")).write_bytes(raw)
        for link in parser.links:
            href=link["href"]
            if href.startswith(("http://","https://")): report.setdefault("discovered_urls",[]).append(href)
            elif href.startswith("/"): report.setdefault("discovered_urls",[]).append(ROOT+href)
            else: report.setdefault("discovered_urls",[]).append(final.rsplit("/",1)[0]+"/"+href)
            if re.search(r"bpp|probe|u_bias|u_current|nistandard|Nidatap|channel|current|voltage",href+" "+link["text"],re.I):
                report.setdefault("matching_links",[]).append({"page":final,**link})
    except Exception as exc:
        report["daq_directory_attempts"].append({"url":das_page,"status":"failed","error":f"{type(exc).__name__}: {exc}"})
urls=list(dict.fromkeys(report.get("discovered_urls", [])))
for ident in identifiers:
    urls.append(f"{ROOT}/utils/data/{SHOT}/{ident}")
    urls.append(f"https://golem.fjfi.cvut.cz/utils/data/{SHOT}/{ident}")
for suffix in ("bpp","lp","U_bias","U_current","probe_voltage","probe_current","PetiProbe:U_bias","PetiProbe:U_current","nistandard","NIstandard","nistandard.txt","bpp.txt","lp.txt"):
    urls.extend([f"{ROOT}/shots/{SHOT}/Diagnostics/PetiProbe/{suffix}",
                 f"{ROOT}/shots/{SHOT}/Diagnostics/PetiProbe/{suffix}.txt",
                 f"{ROOT}/operation/shots/{SHOT}/diagnostics/ParticleFlux/PetiProbe.ON/data/{suffix}"])
urls.extend([
    f"{ROOT}/shots/{SHOT}/DAS/0918TektronixMSO58.ON/main.py",
    f"{ROOT}/shots/{SHOT}/DAS/0918TektronixMSO58.ON/screenshot.png",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/main.py",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/log",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/00_NI%20standard.php",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/Nidatap_6133",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/Nidatap_6133.txt",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/Nidatap_6133.dat",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/Nidatap_6132",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/Nidatap_6132.txt",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/Nidatap_6132.dat",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard.txt",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard.xls",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard6133",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard6133.txt",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard6133.xls",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard6132",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard6132.txt",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard6132.xls",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/bpp",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/bpp.txt",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/11",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/11.txt",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/data",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/data.txt",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/11",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/11.txt",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard",
    f"{ROOT}/shots/{SHOT}/DAS/0512NIstandard.ONN/nistandard.txt",
    f"{ROOT}/shots/{SHOT}/nistandard", f"{ROOT}/shots/{SHOT}/nistandard.txt",
    f"{ROOT}/shots/{SHOT}/NIstandard", f"{ROOT}/shots/{SHOT}/bpp",
    f"{ROOT}/shots/{SHOT}/bpp.txt", f"{ROOT}/shots/{SHOT}/lp",
    f"{ROOT}/shots/{SHOT}/lp.txt",
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/NIstandard.ON/data/nistandard",
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/NIstandard.ON/data/bpp",
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/NIstandard.ON/data/lp",
])
urls=list(dict.fromkeys(urls))

def try_one(url):
    rec={"url":url}
    try:
        raw,final,ctype=get(url)
        rec.update(final_url=final,content_type=ctype,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        decoded = raw.decode("utf-8", "replace")
        if any(url.lower().endswith(x) for x in ("main.py", "/log", ".php", "makefile", "readme", ".json", ".yaml", ".yml")):
            rec["status"] = "text_source"
            rec["text_preview"] = decoded[:5000]
            rec["relevant_lines"] = [line for line in decoded.splitlines()
                if re.search(r"bpp|nistandard|nidatap|data_path|filename|open\(|read\(|write\(|channel|raw", line, re.I)][:80]
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", url.rstrip("/").split("/")[-1])
            (OUT / f"source_{safe}_{hashlib.sha256(url.encode()).hexdigest()[:8]}.txt").write_bytes(raw)
            return rec
        if url.lower().endswith(".npz"):
            try:
                archive = np.load(io.BytesIO(raw), allow_pickle=False)
                arrays = {}
                tvec = np.asarray(archive["time"], dtype=float) if "time" in archive.files else None
                for key in archive.files:
                    value = np.asarray(archive[key])
                    item = {"shape": list(value.shape), "dtype": str(value.dtype)}
                    if tvec is not None and value.ndim == 1 and value.size == tvec.size and key != "time":
                        good = np.isfinite(tvec) & np.isfinite(value)
                        tt, yy = tvec[good], value[good]
                        if tt.size > 100 and np.all(np.diff(tt) > 0):
                            dt = float(np.median(np.diff(tt)))
                            ff = np.fft.rfftfreq(yy.size, dt)
                            pp = np.abs(np.fft.rfft(yy - np.mean(yy)))**2
                            band = (ff >= 1000) & (ff <= min(400e3, 0.45/dt))
                            if np.any(band):
                                inds = np.flatnonzero(band)
                                top = inds[np.argsort(pp[band])[-8:][::-1]]
                                item.update(signal_range=float(np.ptp(yy)), sample_rate_Hz=1/dt,
                                    duration_s=float(tt[-1]-tt[0]),
                                    dominant_frequency_Hz=float(ff[inds[np.argmax(pp[band])]]),
                                    top_frequency_peaks_Hz=[float(ff[z]) for z in top])
                    arrays[key] = item
                rec.update(status="npz_archive", arrays=arrays)
                (OUT / f"archive_{hashlib.sha256(url.encode()).hexdigest()[:8]}.npz").write_bytes(raw)
                return rec
            except Exception as exc:
                rec.update(status="npz_unparsed", parse_error=f"{type(exc).__name__}: {exc}")
                return rec
        try: arr=np.asarray(np.loadtxt(decoded.splitlines()))
        except Exception: arr=np.asarray([])
        if arr.size<10:
            try:
                scalar = float(decoded.strip())
                rec.update(status="numeric_scalar", value=scalar)
            except Exception:
                rec["status"]="empty_or_non_numeric"
        else:
            rec.update(status="numeric_data",shape=list(arr.shape),min=float(np.nanmin(arr)),max=float(np.nanmax(arr)))
            safe=re.sub(r"[^A-Za-z0-9_.-]+","_",url.rstrip("/").split("/")[-1])
            (OUT/f"{safe}_{hashlib.sha256(url.encode()).hexdigest()[:8]}.txt").write_bytes(raw)
            if arr.ndim==2 and arr.shape[1]>=2 and arr.shape[0]>100:
                t=arr[:,0]
                goodt=np.isfinite(t)
                t=t[goodt]
                if t.size>100 and np.all(np.diff(t)>0):
                    dt=float(np.median(np.diff(t)))
                    rec.update(sample_count=int(t.size),sample_rate_Hz=1/dt,duration_s=float(t[-1]-t[0]))
                    channel_spectra=[]
                    for col in range(1,arr.shape[1]):
                        y=arr[:,col]
                        good=np.isfinite(arr[:,0])&np.isfinite(y)
                        yy=y[good]
                        if yy.size<100: continue
                        yy=yy-np.mean(yy)
                        f=np.fft.rfftfreq(yy.size,dt); p=np.abs(np.fft.rfft(yy))**2
                        m=(f>=1000)&(f<=min(400e3,0.45/dt))
                        peak=float(f[m][np.argmax(p[m])]) if np.any(m) else None
                        top_peaks=[]
                        if yy.size>1000:
                            inds=np.flatnonzero(m)
                            top=inds[np.argsort(p[m])[-8:][::-1]] if np.any(m) else []
                            top_peaks=[float(f[z]) for z in top]
                        window_peaks = {}
                        for win_name, lo_t, hi_t in [("preplasma",0.002,0.004),("plasma",0.008,0.015)]:
                            wm = good & (arr[:,0] >= lo_t) & (arr[:,0] <= hi_t)
                            yw = y[wm]
                            if yw.size > 500:
                                yw = yw - np.mean(yw)
                                fw = np.fft.rfftfreq(yw.size, dt)
                                pw = np.abs(np.fft.rfft(yw))**2
                                mw = (fw >= 10000) & (fw <= min(400e3, 0.45/dt))
                                if np.any(mw):
                                    wi = np.flatnonzero(mw)
                                    ti = wi[np.argsort(pw[mw])[-8:][::-1]]
                                    window_peaks[win_name] = [float(fw[z]) for z in ti]
                        channel_spectra.append({"column_index_1based_in_file":col+1,"signal_range":float(np.ptp(y[good])),"dominant_frequency_Hz":peak,"top_frequency_peaks_Hz":top_peaks,"windowed_frequency_peaks_Hz":window_peaks})
                    rec["channel_spectra"]=channel_spectra
                    rec["contains_30_to_70kHz_channel"]=any(30000<=x.get("dominant_frequency_Hz",0)<=70000 for x in channel_spectra)
        return rec
    except Exception as exc: return {"url":url,"status":"failed","error":f"{type(exc).__name__}: {exc}"}

with ThreadPoolExecutor(max_workers=24) as pool:
    report["candidate_results"]=list(pool.map(try_one,urls))
numeric=[x for x in report["candidate_results"] if x["status"]=="numeric_data"]
report["numeric_candidate_count"]=len(numeric)
report["possible_50kHz_sweeps"]=[
    {"url":x["url"],"channels":[c for c in x.get("channel_spectra",[]) if 30000<=c.get("dominant_frequency_Hz",0)<=70000]}
    for x in numeric if x.get("contains_30_to_70kHz_channel")
]
report["direct_Ti_fit_possible_from_found_channels"]=any("U_bias" in x["url"] for x in numeric) and any("U_current" in x["url"] for x in numeric)
report["high_speed_scope_files_found"]=[x for x in numeric if "data_all" in x["url"].lower()]
report["interpretation"]="A 50 kHz component in a scope file is only a sweep candidate; channel labels must identify bias and current before Ti fitting."
(OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
