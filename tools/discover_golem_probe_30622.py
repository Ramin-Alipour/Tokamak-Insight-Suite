#!/usr/bin/env python3
"""Discover GOLEM shot 30622 probe signals from Data.php and known data APIs."""
from __future__ import annotations
import hashlib, json, re
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

pages=[f"{ROOT}/shots/{SHOT}/Data.php",f"{ROOT}/shots/{SHOT}/",f"https://golem.fjfi.cvut.cz/shots/{SHOT}/Data.php"]
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
            joined=link["href"]+" "+link["text"]
            if re.search(r"bpp|ball.?pen|probe|nistandard|u_bias|u_current|sweep|langmuir",joined,re.I):
                report.setdefault("matching_links",[]).append({"page":final,**link})
                href=link["href"]
                tail=href.rstrip("/").split("/")[-1]
                if re.fullmatch(r"[A-Za-z0-9_:/.-]{2,100}",tail): identifiers.add(tail)
        (OUT/f"page_{len(report['page_attempts']):02d}.html").write_bytes(raw)
    except Exception as exc:
        rec.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    report["page_attempts"].append(rec)

urls=[]
for ident in identifiers:
    urls.append(f"{ROOT}/utils/data/{SHOT}/{ident}")
    urls.append(f"https://golem.fjfi.cvut.cz/utils/data/{SHOT}/{ident}")
for suffix in ("bpp","lp","U_bias","U_current","probe_voltage","probe_current","PetiProbe:U_bias","PetiProbe:U_current"):
    urls.extend([f"{ROOT}/shots/{SHOT}/Diagnostics/PetiProbe/{suffix}",
                 f"{ROOT}/shots/{SHOT}/Diagnostics/PetiProbe/{suffix}.txt",
                 f"{ROOT}/operation/shots/{SHOT}/diagnostics/ParticleFlux/PetiProbe.ON/data/{suffix}"])
urls=list(dict.fromkeys(urls))

def try_one(url):
    rec={"url":url}
    try:
        raw,final,ctype=get(url)
        rec.update(final_url=final,content_type=ctype,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        try: arr=np.asarray(np.loadtxt(raw.decode("utf-8","replace").splitlines()))
        except Exception: arr=np.asarray([])
        if arr.size<10: rec["status"]="empty_or_non_numeric"
        else:
            rec.update(status="numeric_data",shape=list(arr.shape),min=float(np.nanmin(arr)),max=float(np.nanmax(arr)))
            safe=re.sub(r"[^A-Za-z0-9_.-]+","_",url.rstrip("/").split("/")[-1])
            (OUT/f"{safe}_{hashlib.sha256(url.encode()).hexdigest()[:8]}.txt").write_bytes(raw)
            if arr.ndim==2 and arr.shape[1]>=2 and arr.shape[0]>100:
                t,y=arr[:,0],arr[:,1]; good=np.isfinite(t)&np.isfinite(y); t,y=t[good],y[good]
                if t.size>100 and np.all(np.diff(t)>0):
                    dt=float(np.median(np.diff(t))); rec.update(sample_count=int(t.size),sample_rate_Hz=1/dt,duration_s=float(t[-1]-t[0]),signal_range=float(np.ptp(y)))
                    if t.size>1000:
                        f=np.fft.rfftfreq(t.size,dt); p=np.abs(np.fft.rfft(y-np.mean(y)))**2
                        m=(f>=1000)&(f<=min(2e6,0.45/dt))
                        if np.any(m): rec["dominant_frequency_Hz"]=float(f[m][np.argmax(p[m])])
        return rec
    except Exception as exc: return {"url":url,"status":"failed","error":f"{type(exc).__name__}: {exc}"}

with ThreadPoolExecutor(max_workers=24) as pool:
    report["candidate_results"]=list(pool.map(try_one,urls))
numeric=[x for x in report["candidate_results"] if x["status"]=="numeric_data"]
report["numeric_candidate_count"]=len(numeric)
report["possible_50kHz_sweeps"]=[x for x in numeric if 30000<=x.get("dominant_frequency_Hz",0)<=70000]
report["direct_Ti_fit_possible_from_found_channels"]=any("U_bias" in x["url"] for x in numeric) and any("U_current" in x["url"] for x in numeric)
(OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
