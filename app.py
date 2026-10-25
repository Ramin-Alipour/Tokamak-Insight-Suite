from __future__ import annotations
import json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import streamlit as st
from tis import __version__
from tis.ui.catalog_page import render_catalog_page
from tis.ui.project_page import render_project_page
from tokamak_analysis.analysis_registry import ANALYSES, resolve_analysis
from tokamak_analysis.molecular_spectroscopy import analyze_molecular_spectrum, fit_custom_molecular_model, fit_molecular_boltzmann_temperature
from tokamak_analysis.cxrs import analyze_cxrs_line
from tokamak_analysis.poloidal_field_probes import analyze_poloidal_field_probes
from tokamak_analysis.thomson_spectrum import fit_thomson_spectrum
from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution
from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power
from tokamak_analysis.thomson_workflow import analyze_thomson_profile
from tokamak_analysis.ece_analysis import analyze_ece
from tis.catalog import DIAGNOSTICS, FORMULAS, diagnostic_families, maturity_counts, maturity_label
from tis.signals import sampling_audit, hampel_despike, prepare_time_series_robust
from tis.events import detect_elms, detect_sawteeth, current_quench_analysis
from tis.plasma import (confinement_summary, beta_metrics, collisionality_nu_star, spitzer_resistivity,
    larmor_radii_and_speeds, coulomb_logarithm_electron, shaping_surface_area, q95_iter_formula)
from tis.report import build_html_report, figure_to_png_base64
from tis.contributions import build_contribution_bundle, bundle_json
from tis.data_sources import fetch_fair_mast_signal, fetch_fair_mast_efit_snapshot, list_fair_mast_signals, fetch_public_table_bytes
from tis.core import (load_table, signal_quality, sha256_bytes, linear_calibration, polynomial_calibration,
    prepare_time_series, butterworth_filter, fft_spectrum, welch_psd, cross_analysis, cross_correlation_lag,
    fit_spectral_line, doppler_temperature_eV, doppler_temperature_with_uncertainty_eV, doppler_velocity_with_uncertainty_m_s, doppler_velocity_m_s, derive_parameters,
    uncertainty_linear, grid_from_table, regularized_tomography, select_tikhonov_lambda, spectral_calibration, doppler_temperature_from_fwhm_eV, probe_iv_summary,
    linear_phase_density, langmuir_iv_analysis, mode_number_from_phase, mode_number_from_channels, fluctuation_flux,
    exponential_heat_flux_fit, eich_heat_flux_fit, integrate_profile, spectrum_moments, neutron_rate_from_counts,
    equilibrium_grid_derivatives, xray_spectrum_summary, abel_reconstruction, axisymmetric_radiated_power, profile_gradient_analysis, triple_probe_temperature_eV, integrate_time_series, reflectometry_cutoff_density, electron_cyclotron_resonance_field, corrected_line_ratio, time_frequency_spectrogram, autocorrelation_analysis)

st.set_page_config(page_title='Tokamak Insight Suite', page_icon='⚛️', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
:root{--navy:#07182b;--panel:#0b2138;--panel2:#102b46;--blue:#168bff;--line:#204463;--text:#e6f1ff;}
.stApp{background:linear-gradient(135deg,#061426 0%,#081b30 55%,#07172a 100%);color:var(--text)}
[data-testid="stSidebar"]{background:#07192b;border-right:1px solid #163a5a}
[data-testid="stMetric"]{background:linear-gradient(145deg,#102b46,#0b2138);border:1px solid #204463;padding:15px;border-radius:12px}
[data-testid="stMetricLabel"]{color:#9fc7ec!important}[data-testid="stMetricValue"]{color:#f1f7ff!important}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:#204463!important;border-radius:12px!important}
.stButton>button,.stDownloadButton>button{background:#168bff;color:white;border:0;border-radius:8px;font-weight:600}
.stButton>button:hover,.stDownloadButton>button:hover{background:#3ba0ff;color:white}
section[data-testid="stFileUploaderDropzone"]{background:#0c2238;border:1px dashed #2b6590}
hr{border-color:#204463}
h1,h2,h3{color:#f0f7ff!important}
.small-muted{color:#9fc7ec;font-size:.88rem}
</style>''', unsafe_allow_html=True)

def _clear_loaded_data(clear_uploader=True):
    for key in ("tis_raw", "tis_filename", "tis_df", "tis_sha", "tis_source_manifest",
                "tis_upload_sha", "tis_upload_signature", "tis_upload_identity",
                "tis_upload_raw", "tis_active_source", "tis_imported_at_utc",
                "tis_import_parameters", "tis_analysis_history", "tis_report_sections", "tis_project_export_bytes", "tis_project_export_signature"):
        st.session_state.pop(key, None)
    # Widget state may only be changed before its widget is instantiated in a run.
    # The Clear button invokes this as a callback (before widget creation); the
    # automatic stale-upload cleanup below runs after file_uploader was created.
    if clear_uploader:
        st.session_state["global_upload"] = None

# Sidebar: project identity and persistent data selection.
with st.sidebar:
    st.markdown('## ⚛️ Tokamak Insight Suite')
    st.caption('Data Analysis · Diagnostics · Plasma Physics')
    st.info('OPEN RESEARCH BETA · Scientific validation is not a prerequisite for using or contributing to TIS. Share reproducible cases and report issues; results remain the researcher’s responsibility.')
    st.divider()
    project=st.text_input('Project / experiment',st.session_state.get('project','Untitled tokamak analysis'),key='project')
    shot=st.text_input('Shot / discharge ID',st.session_state.get('shot',''),key='shot')
    analyst=st.text_input('Analyst',st.session_state.get('analyst',''),key='analyst')
    st.markdown('**Load diagnostic data**')
    upload=st.file_uploader('CSV · TXT · Excel · Parquet',type=['csv','txt','xlsx','xls','parquet','pq'],key='global_upload')
    with st.expander("Import parsing options"):
        import_skiprows=st.number_input("Rows to skip before header",min_value=0,max_value=10000,value=0,step=1,key="import_skiprows")
        import_decimal=st.selectbox("Decimal separator",[".",","],index=0,key="import_decimal")
        import_delimiter=st.selectbox("CSV/TXT delimiter",["Auto-detect",",",";","Tab","|"],index=0,key="import_delimiter")
        import_comment=st.selectbox("Comment marker",["None","#","%"],index=0,key="import_comment")
        import_sheet=st.text_input("Excel sheet name (use index:N for index; blank = first sheet)",value="",key="import_sheet")
        import_encoding=st.selectbox("Text encoding",["Auto (UTF-8 → Windows-1252 → Latin-1)","utf-8","cp1252","latin-1","utf-16"],index=0,key="import_encoding")
        import_drop_units=st.checkbox("Drop first data row if it contains units (e.g. V / s)",value=False,key="import_drop_units")
    if upload is None and st.session_state.get("tis_active_source") == "upload":
        # The uploader was cleared; do not keep silently analysing the previous upload.
        # Its widget already exists in this run, so do not mutate its session-state key.
        _clear_loaded_data(clear_uploader=False)
    if upload is not None:
        try:
            upload_identity=f"{getattr(upload,'file_id',upload.name)}:{upload.name}:{upload.size}"
            new_upload=upload_identity != st.session_state.get("tis_upload_identity")
            if new_upload:
                uploaded_bytes=upload.getvalue()
                uploaded_sha=sha256_bytes(uploaded_bytes)
            else:
                uploaded_bytes=st.session_state.get("tis_upload_raw",b"")
                uploaded_sha=st.session_state.get("tis_upload_sha","")
            delimiter_map={"Auto-detect":None,",":",",";":";","Tab":"\t","|":"|"}
            sheet_text=import_sheet.strip()
            if not sheet_text:
                sheet_value=0
            elif sheet_text.lower().startswith("index:"):
                index_text=sheet_text.split(":",1)[1].strip()
                if not index_text.isdigit():
                    raise ValueError("Excel sheet index must use index:N with a nonnegative integer, e.g. index:0.")
                sheet_value=int(index_text)
            else:
                # Keep numeric-looking worksheet names (e.g. "2024") as names.
                sheet_value=sheet_text
            import_signature=uploaded_sha+json.dumps({"delimiter":import_delimiter,"decimal":import_decimal,
                "skiprows":int(import_skiprows),"comment":import_comment,"sheet_name":sheet_value,"drop_units_row":bool(import_drop_units),"encoding":import_encoding},sort_keys=True)
            needs_import=(new_upload or st.session_state.get("tis_active_source")!="upload" or
                import_signature != st.session_state.get("tis_upload_signature"))
            if needs_import:
                frame=load_table(uploaded_bytes,upload.name,sheet_name=sheet_value,
                                 delimiter=delimiter_map[import_delimiter],decimal=import_decimal,
                                 skiprows=int(import_skiprows),comment=None if import_comment=="None" else import_comment,
                                 encoding=None if import_encoding.startswith("Auto") else import_encoding)
                if import_drop_units:
                    if len(frame)<2: raise ValueError("Cannot drop a units row because the table has fewer than two rows.")
                    frame=frame.iloc[1:].reset_index(drop=True)
                    for column in frame.columns:
                        converted=pd.to_numeric(frame[column],errors="coerce")
                        # Keep a column numeric only if the conversion lost no non-empty cell.
                        if int(converted.notna().sum())==int(frame[column].notna().sum()): frame[column]=converted
                st.session_state.pop("tis_source_manifest",None)
                st.session_state["tis_raw"]=uploaded_bytes
                st.session_state["tis_filename"]=upload.name
                st.session_state["tis_df"]=frame
                st.session_state["tis_sha"]=uploaded_sha
                st.session_state["tis_upload_sha"]=uploaded_sha
                st.session_state["tis_upload_signature"]=import_signature
                st.session_state["tis_upload_identity"]=upload_identity
                st.session_state["tis_upload_raw"]=uploaded_bytes
                st.session_state.pop("tis_analysis_history",None)
                # Report sections belong to the dataset that produced them; never carry them into a new import.
                st.session_state.pop("tis_report_sections", None)
                st.session_state.pop("tis_project_export_bytes", None)
                st.session_state.pop("tis_project_export_signature", None)
                st.session_state["tis_active_source"]="upload"
                st.session_state["tis_imported_at_utc"]=datetime.now(timezone.utc).isoformat()
                st.session_state["tis_import_parameters"]={"delimiter":import_delimiter,"decimal":import_decimal,
                    "skiprows":int(import_skiprows),"comment":import_comment,"sheet_name":sheet_value,"drop_units_row":bool(import_drop_units),
                    "encoding_requested":import_encoding,"encoding_used":frame.attrs.get("encoding")}
        except Exception as e:
            # Never leave a previous dataset active after a failed import attempt:
            # the user may otherwise analyse stale data while believing the new
            # file or parsing configuration was accepted.
            for key in ("tis_raw", "tis_filename", "tis_df", "tis_sha", "tis_source_manifest",
                        "tis_upload_sha", "tis_upload_signature", "tis_upload_identity",
                        "tis_upload_raw", "tis_active_source", "tis_imported_at_utc",
                        "tis_import_parameters", "tis_analysis_history", "tis_report_sections",
                        "tis_project_export_bytes", "tis_project_export_signature"):
                st.session_state.pop(key, None)
            st.error(f"Import failed; no dataset is active. {e}")
    if 'tis_df' in st.session_state:
        st.success(f"Loaded: {st.session_state.get('tis_filename','data')}")
        st.button('Clear loaded data',use_container_width=True,on_click=_clear_loaded_data)
    st.divider()
    pages=['Home','Public Data Access','Data Import & QC','Time Series & MHD','Events & Transients','Spectral Analysis','Probe Analysis','Plasma Parameters','Confinement & Scalings','Equilibrium & Reconstruction','Tomography & Bolometry','Advanced Analysis','Diagnostic Catalog','Project Manager','Reports & Export']
    requested=st.session_state.pop('jump','Home')
    page=st.radio('WORKSPACE',pages,index=pages.index(requested) if requested in pages else 0,key='workspace_page',label_visibility='collapsed')
    st.divider()
    st.caption('Reference catalog: extracted tokamak diagnostic reference')
    st.caption(f'Independent Python project · v{__version__}')

df=st.session_state.get('tis_df')
filename=st.session_state.get('tis_filename','')
raw=st.session_state.get('tis_raw',b'')
sha=st.session_state.get('tis_sha','')
numeric=[] if df is None else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def col_select(label, options, key, default=None):
    if not options: st.warning('No numeric columns are available. Load a suitable data file first.'); st.stop()
    idx=options.index(default) if default in options else 0
    return st.selectbox(label,options,index=idx,key=key)
def plot_line(x,y,xlabel,ylabel,title=None,logy=False):
    # Plot a bounded number of points for responsive rendering on million-sample shots.
    xx=np.asarray(x).ravel(); yy=np.asarray(y).ravel()
    if len(xx)!=len(yy): raise ValueError('Plot x/y arrays must have equal lengths.')
    if len(xx)>20000:
        # Bucketed min/max decimation preserves ELM-like spikes and glitches.
        edges=np.linspace(0,len(xx),10001,dtype=int)
        keep=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            if hi>lo:
                segment=yy[lo:hi]
                finite_idx=np.flatnonzero(np.isfinite(segment))
                if finite_idx.size:
                    finite_values=segment[finite_idx]
                    keep.extend((
                        lo+int(finite_idx[int(np.argmin(finite_values))]),
                        lo+int(finite_idx[int(np.argmax(finite_values))]),
                    ))
                    invalid_idx=np.flatnonzero(~np.isfinite(segment))
                    if invalid_idx.size:
                        # Retain a non-finite point so the plot does not
                        # silently connect across a missing/invalid region.
                        keep.append(lo+int(invalid_idx[0]))
                else:
                    # Preserve an all-invalid bucket as a gap rather than
                    # allowing NaNs to mask extrema in neighbouring buckets.
                    keep.append(lo)
        idx=np.asarray(sorted(set(keep)),dtype=int)
        xx,yy=xx[idx],yy[idx]
    fig=Figure(figsize=(9,3.3)); ax=fig.subplots(); ax.plot(xx,yy,lw=1.0)
    if logy: ax.set_yscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig)
def _json_safe(value):
    if isinstance(value, dict): return {str(k): _json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return _json_safe(value.tolist())
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (complex, np.complexfloating)):
        return {
            "real": float(value.real) if np.isfinite(value.real) else None,
            "imag": float(value.imag) if np.isfinite(value.imag) else None,
        }
    if isinstance(value, np.floating): return float(value) if np.isfinite(value) else None
    if isinstance(value, float): return value if np.isfinite(value) else None
    if isinstance(value, np.bool_): return bool(value)
    return value

def _result_summary(value):
    if isinstance(value, dict):
        return {str(k): _result_summary(v) for k,v in value.items()
                if not isinstance(v,np.ndarray) or v.size<=100}
    if isinstance(value,np.ndarray):
        finite=value[np.isfinite(value)] if np.issubdtype(value.dtype,np.number) else np.asarray([])
        summary={"shape":list(value.shape),"dtype":str(value.dtype),"finite_count":int(finite.size)}
        if finite.size and np.iscomplexobj(finite):
            # Complex-valued spectra/cross-spectra have no natural scalar min/max.
            # Preserve their components and magnitude instead of raising TypeError
            # when float(complex) is attempted during report-history generation.
            summary.update({
                "real_min":float(np.min(finite.real)),
                "real_max":float(np.max(finite.real)),
                "imag_min":float(np.min(finite.imag)),
                "imag_max":float(np.max(finite.imag)),
                "magnitude_min":float(np.min(np.abs(finite))),
                "magnitude_max":float(np.max(np.abs(finite))),
            })
        else:
            summary.update({
                "min":float(np.min(finite)) if finite.size else None,
                "max":float(np.max(finite)) if finite.size else None,
            })
        return summary
    if isinstance(value,(list,tuple)) and len(value)>30:
        return {"items":len(value),"preview":_json_safe(value[:5])}
    return _json_safe(value)

def download_df(label, data, name):
    st.download_button(label,data.to_csv(index=False).encode(),name,'text/csv')

def _parameter_snapshot():
    snapshot={}
    for key,value in st.session_state.items():
        if key.startswith("tis_") or key in {"global_upload","tomo_A","tomo_y","tomo_sigma","rad_E","tomo_ray_start","tomo_ray_end","rad_R","rad_Z"}:
            continue
        if isinstance(value,(str,int,float,bool,type(None),list,tuple,dict,np.integer,np.floating,np.bool_)):
            try: snapshot[str(key)]=_json_safe(value)
            except (TypeError,ValueError): continue
    return snapshot

def _record_download_history(name, digest, result_summary, page_name, parameter_snapshot):
    """Record an analysis export only after the user actually clicks its download button."""
    history=st.session_state.get("tis_analysis_history",[])
    provenance = {
        "source_filename": st.session_state.get("tis_filename"),
        "source_sha256": st.session_state.get("tis_sha"),
        "shot_id": st.session_state.get("shot"),
        "source_manifest": _json_safe(st.session_state.get("tis_source_manifest")),
        "import_parameters": _json_safe(st.session_state.get("tis_import_parameters")),
    }
    if any(
        h.get("name") == name
        and h.get("result_sha256") == digest
        and h.get("parameter_snapshot") == parameter_snapshot
        and all(h.get(key) == value for key, value in provenance.items())
        for h in history
    ):
        return
    history=[{"name":name,"created_at_utc":datetime.now(timezone.utc).isoformat(),
              "page":page_name,"parameter_snapshot":parameter_snapshot,"result_sha256":digest,
              "result_summary":result_summary, **provenance}]+history
    st.session_state["tis_analysis_history"]=history[:50]
    st.session_state.pop("tis_project_export_bytes", None)
    st.session_state.pop("tis_project_export_signature", None)

def json_download(label,obj,name):
    safe=_json_safe(obj)
    payload=json.dumps(safe,indent=2,ensure_ascii=False,allow_nan=False).encode()
    if any(token in name.lower() for token in ("provenance","registry","formula_registry")):
        st.download_button(label,payload,name,'application/json')
    else:
        import hashlib
        digest=hashlib.sha256(payload).hexdigest()
        st.download_button(
            label,payload,name,'application/json',
            on_click=_record_download_history,
            args=(name,digest,_result_summary(obj),page,_parameter_snapshot()),
        )
def add_report_section(title, results, fig=None, notes=""):
    """Remember a result (and optional figure) so it appears in the HTML report. Replaces any previous section of the same title."""
    sections=[x for x in st.session_state.get('tis_report_sections',[]) if x['title']!=title]
    sections.append({'title':title,'results':_json_safe(results),'notes':notes,
                     'figures':[figure_to_png_base64(fig)] if fig is not None else []})
    st.session_state['tis_report_sections']=sections[-30:]
def need_data():
    if df is None:
        st.info('Load a diagnostic data file using the left-hand panel. The diagnostic library and formula registry remain available without a data file.')
        st.stop()

@st.cache_data(show_spinner=False, max_entries=3)
def cached_prepare_time_series(time_values, signal_values, sample_rate, gap_policy="error"):
    t_, y_, fs_, report_ = prepare_time_series_robust(time_values, signal_values, sample_rate, gap_policy)
    return t_, y_, fs_, report_

@st.cache_data(show_spinner=False, max_entries=3)
def cached_filter(values, fs, highpass, lowpass, order):
    return butterworth_filter(values, fs, highpass, lowpass, order)

@st.cache_data(show_spinner=False, max_entries=3)
def cached_fft(values, fs, window):
    return fft_spectrum(values, fs, window=window)

@st.cache_data(show_spinner=False, max_entries=3)
def cached_welch(values, fs, nperseg, noverlap, window):
    return welch_psd(values, fs, nperseg, noverlap, window)

if page=='Home':
    st.markdown('# Tokamak Insight Suite')
    st.markdown('**Integrated Analysis Platform for Tokamak Diagnostics**')
    st.markdown('From raw diagnostic data to physical insight — import, inspect, analyze, calibrate, reconstruct and export results in one Python-based workspace.')
    a,b,c=st.columns([1.2,1,1])
    with a:
        st.markdown('### Project workspace')
        st.write(f'**Project:** {project}')
        st.write(f'**Shot / discharge:** {shot or "Not specified"}')
        st.write(f'**Input:** {filename or "No data loaded"}')
        st.write('**Data integrity:** '+(sha[:16]+'…' if sha else 'Not available'))
        if st.button('Open Data Import & QC',use_container_width=True): st.session_state['jump']='Data Import & QC'; st.rerun()
    with b:
        st.metric('Diagnostic entries',len(DIAGNOSTICS)); st.metric('Formula references',len(FORMULAS))
    with c:
        st.metric('Diagnostic families',len(diagnostic_families())); st.metric('Supported table formats','6')
    st.divider()
    left,right=st.columns([1.7,1])
    with left:
        st.markdown('### Quick preview')
        if df is not None and numeric:
            pcol=st.selectbox('Preview channel',numeric,key='home_preview_col')
            if len(df)>1: plot_line(np.arange(len(df)),pd.to_numeric(df[pcol],errors='coerce'),'Sample index',pcol,'Loaded data preview')
            st.dataframe(df.head(8),use_container_width=True)
        else:
            st.info('Upload a CSV, TXT, Excel or Parquet file from the left panel to activate quick preview and analysis tools.')
            st.markdown('#### Quick tools')
            st.write('📂 Load Data  ·  〰️ Quick Analysis  ·  🧮 Plasma Parameters  ·  📚 Diagnostic Catalog')
    with right:
        st.markdown('### Quick tools')
        for label,dest in [('Load Data','Data Import & QC'),('Quick Analysis','Time Series & MHD'),('Calculate Plasma Parameters','Plasma Parameters'),('Open Diagnostic Catalog','Diagnostic Catalog')]:
            if st.button(label,use_container_width=True,key='quick_'+dest): st.session_state['jump']=dest; st.rerun()
        st.markdown('### Recent activity')
        for item in st.session_state.get('tis_activity',[])[:5]: st.caption('• '+item)
    st.markdown('### Diagnostic categories')
    categories=['Magnetic & Equilibrium','Probes','Spectroscopy','Imaging & Tomography','Neutrons & Gamma','MHD & Fluctuations','Heating & RF','Wall & Materials','Interferometry & Density','Advanced Analysis','Data Processing','Utilities']
    grid=st.columns(4)
    for i,x in enumerate(categories): grid[i%4].markdown(f'**{x}**')
    st.caption('Scientific outputs must be interpreted with the actual diagnostic calibration, units, geometry, assumptions and uncertainty model.')

elif page=='Public Data Access':
    st.title('Public Experimental Data Access')
    st.caption('Fetch a single time-series signal directly from the public UKAEA FAIR-MAST archive. Shot/group availability varies; this does not download the entire archive.')
    st.markdown('Source documentation: [FAIR-MAST Level-1 data](https://mastapp.site/level1-data.html) · [Archive overview](https://www.ukaea.org/service/fair-mast/)')
    st.subheader('Discover signals in a public shot')
    with st.form('fair_mast_inventory_form'):
        inv1, inv2 = st.columns(2)
        with inv1:
            inventory_shot = st.number_input('Shot to inspect', min_value=1, value=30421, step=1, key='inventory_shot')
        with inv2:
            inventory_group = st.selectbox('Group to inspect', ['amc', 'ane', 'ayc', 'efm', 'xmo'], key='inventory_group', help='Some groups contain multidimensional profiles or images that need a dedicated loader.')
        inventory_clicked = st.form_submit_button('List available signals')
    if inventory_clicked:
        try:
            with st.spinner('Reading signal metadata from FAIR-MAST...'):
                inventory_frame, inventory_manifest = list_fair_mast_signals(int(inventory_shot), inventory_group)
                st.session_state['tis_signal_inventory'] = inventory_frame
                st.session_state['tis_signal_inventory_manifest'] = inventory_manifest
        except Exception as e:
            st.error(f'Signal inventory failed: {e}')
    if 'tis_signal_inventory' in st.session_state:
        inventory_frame = st.session_state['tis_signal_inventory']
        inventory_manifest = st.session_state['tis_signal_inventory_manifest']
        st.success(f"{inventory_manifest['variable_count']} variables found · {inventory_manifest['one_dimensional_numeric_time_traces']} numeric 1-D time traces supported by the current importer")
        st.dataframe(inventory_frame, use_container_width=True, hide_index=True)
        st.download_button('Download signal inventory CSV', data=inventory_frame.to_csv(index=False).encode('utf-8'), file_name=f"mast_{inventory_manifest['shot']}_{inventory_manifest['source_group']}_signal_inventory.csv", mime='text/csv')
        with st.expander('Inventory provenance and limitations'):
            st.json(inventory_manifest)

    with st.form('fair_mast_fetch_form'):
        c1, c2 = st.columns(2)
        with c1:
            mast_shot = st.number_input('MAST shot number', min_value=1, value=29790, step=1)
            mast_group = st.selectbox('Diagnostic group', ['xmo', 'amc', 'ane'], help='xmo: Mirnov coils; amc: currents; ane: interferometry. Not every group exists in every shot.')
        with c2:
            mast_signal = st.text_input('Signal name (leave blank to auto-select)', value='', help='For xmo, the default is omaha_1lz when available.')
            mast_max_samples = st.number_input('Maximum exported samples', min_value=1024, max_value=500000, value=70000, step=1024)
        fetch_clicked = st.form_submit_button('Fetch public signal')
    if fetch_clicked:
        try:
            with st.spinner('Connecting to FAIR-MAST and reading the selected signal...'):
                mast_frame, mast_manifest = fetch_fair_mast_signal(
                    int(mast_shot), mast_group, mast_signal.strip() or None, int(mast_max_samples)
                )
                mast_csv = mast_frame.to_csv(index=False).encode('utf-8')
                mast_filename = f"mast_{int(mast_shot)}_{mast_group}_{mast_manifest['signal']}.csv"
                st.session_state.pop('tis_source_manifest', None)
                st.session_state['tis_active_source'] = 'mast'
                st.session_state.pop('tis_analysis_history', None)
                st.session_state.pop('tis_project_export_bytes', None)
                st.session_state.pop('tis_project_export_signature', None)
                st.session_state['tis_raw'] = mast_csv
                st.session_state['tis_filename'] = mast_filename
                st.session_state['tis_imported_at_utc'] = datetime.now(timezone.utc).isoformat()
                st.session_state['tis_df'] = mast_frame
                st.session_state['tis_sha'] = sha256_bytes(mast_csv)
                st.session_state['tis_source_manifest'] = mast_manifest
                st.session_state['tis_source_manifest']['imported_at_utc'] = st.session_state['tis_imported_at_utc']
                st.session_state['tis_source_manifest']['csv_sha256'] = st.session_state['tis_sha']
                st.session_state['tis_source_manifest']['csv_file'] = mast_filename
            st.rerun()
        except Exception as e:
            st.error(f'Public data fetch failed: {e}')
            st.info('Check the shot/group/signal name and internet access. Some public archives or individual groups may be temporarily unavailable.')
    if 'tis_source_manifest' in st.session_state:
        manifest = st.session_state['tis_source_manifest']
        st.success(f"Loaded shot {manifest['shot']} · {manifest['source_group']} / {manifest['signal']} · {manifest['exported_samples']:,} samples")
        st.caption(f"Source class: {manifest['source_signal_class']} · Quality: {manifest['source_quality']} · Units: {manifest['source_units']}")
        st.dataframe(st.session_state['tis_df'].head(30), use_container_width=True)
        st.json(manifest)
        st.download_button('Download CSV', data=st.session_state['tis_raw'], file_name=st.session_state['tis_filename'], mime='text/csv')
        st.download_button('Download provenance manifest', data=json.dumps(manifest, indent=2), file_name='mast_source_manifest.json', mime='application/json')
        st.caption('The fetched data are not calibrated by TIS. Verify units, calibration and quality before physical interpretation.')

    st.markdown('---')
    st.subheader('Import a direct public data-file URL')
    st.caption('For archives that provide direct CSV/TXT/Parquet exports, such as GOLEM. This downloads a single file; it does not crawl or scrape websites.')
    with st.form('public_table_url_form'):
        public_url = st.text_input(
            'Direct data-file URL',
            value='',
            placeholder='https://golem.fjfi.cvut.cz/utils/data/23078/loop_voltage.csv',
            help='Use a direct file link, not a shot homepage or catalog page. Supported formats: CSV, TXT, XLSX/XLS and Parquet.',
        )
        public_max_mb = st.number_input('Maximum download size (MB)', min_value=1, max_value=100, value=25, step=1)
        public_fetch_clicked = st.form_submit_button('Download and import URL')
    if public_fetch_clicked:
        try:
            with st.spinner('Downloading public data file...'):
                payload, source_meta = fetch_public_table_bytes(
                    public_url, max_bytes=int(public_max_mb) * 1_000_000
                )
                from urllib.parse import urlparse
                public_filename = Path(urlparse(public_url).path).name or 'public_data.csv'
                if Path(public_filename).suffix.lower() not in {'.csv', '.txt', '.xlsx', '.xls', '.parquet', '.pq'}:
                    public_filename += '.txt'
                public_frame = load_table(payload, public_filename)
                st.session_state.pop('tis_source_manifest', None)
                st.session_state['tis_active_source'] = 'public_url'
                st.session_state.pop('tis_analysis_history', None)
                st.session_state.pop('tis_project_export_bytes', None)
                st.session_state.pop('tis_project_export_signature', None)
                st.session_state['tis_raw'] = payload
                st.session_state['tis_filename'] = public_filename
                st.session_state['tis_imported_at_utc'] = datetime.now(timezone.utc).isoformat()
                st.session_state['tis_df'] = public_frame
                st.session_state['tis_sha'] = sha256_bytes(payload)
                st.session_state['tis_source_manifest'] = {
                    'dataset': 'Public direct-file URL',
                    'shot': 'As encoded in source URL / metadata',
                    'source_group': 'Direct tabular file',
                    'signal': ', '.join(map(str, public_frame.columns)),
                    'source_signal_class': 'As published by source',
                    'source_quality': 'Not independently assessed',
                    'source_units': 'Preserved as provided; inspect source documentation',
                    'imported_at_utc': st.session_state['tis_imported_at_utc'],
                    'csv_sha256': st.session_state['tis_sha'],
                    'rows': int(len(public_frame)),
                    'columns': [str(x) for x in public_frame.columns],
                    **source_meta,
                }
            st.rerun()
        except Exception as e:
            st.error(f'Public URL import failed: {e}')
            st.info('Use a direct file URL and confirm that the source permits public download.')


    st.markdown('---')
    st.subheader('Fetch an existing EFIT equilibrium grid')
    st.caption('Loads the archive’s existing EFIT-derived plasma-current R–Z grid. This is an archive export, not a new equilibrium reconstruction or Grad–Shafranov solver.')
    with st.form('fair_mast_efit_form'):
        ef1, ef2 = st.columns(2)
        with ef1:
            efit_shot = st.number_input('EFIT shot number', min_value=1, value=30421, step=1, key='efit_shot')
        with ef2:
            efit_time = st.number_input('Requested time (s)', value=0.2, format='%.6f', key='efit_time')
        efit_fetch_clicked = st.form_submit_button('Fetch EFIT grid')
    if efit_fetch_clicked:
        try:
            with st.spinner('Reading the public EFIT R–Z grid...'):
                efit_frame, efit_manifest = fetch_fair_mast_efit_snapshot(int(efit_shot), float(efit_time))
                st.session_state['tis_efit_frame'] = efit_frame
                st.session_state['tis_efit_manifest'] = efit_manifest
        except Exception as e:
            st.error(f'EFIT grid fetch failed: {e}')
            st.info('Confirm that this shot has an accessible efm group and that public-data dependencies are installed.')
    if 'tis_efit_frame' in st.session_state and 'tis_efit_manifest' in st.session_state:
        efit_frame = st.session_state['tis_efit_frame']
        efit_manifest = st.session_state['tis_efit_manifest']
        st.success(f"Loaded shot {efit_manifest['shot']} · EFIT grid {efit_manifest['grid_shape_z_by_r']} at t={efit_manifest['selected_time_s']:.6f} s")
        em1, em2, em3 = st.columns(3)
        em1.metric('Grid points', f"{efit_frame.shape[0] * (efit_frame.shape[1] - 1):,}")
        em2.metric('Finite grid fraction', f"{efit_manifest['finite_grid_fraction']:.1%}")
        em3.metric('Archive quality', str(efit_manifest['source_quality']))
        try:
            r_coords = np.asarray([float(str(col)[2:-2]) for col in efit_frame.columns[1:]], dtype=float)
            z_coords = efit_frame['z_m'].to_numpy(dtype=float)
            grid_values = efit_frame.iloc[:, 1:].to_numpy(dtype=float)
            fig=Figure(figsize=(8,4.5)); ax=fig.subplots()
            mesh = ax.pcolormesh(r_coords, z_coords, grid_values, shading='auto')
            ax.set_xlabel('R coordinate (m)'); ax.set_ylabel('Z coordinate (m)')
            ax.set_title(f"Existing EFIT plasma-current grid · shot {efit_manifest['shot']} · t={efit_manifest['selected_time_s']:.4f} s")
            fig.colorbar(mesh, ax=ax, label='Archived plasma_current_rz value')
            fig.tight_layout(); st.pyplot(fig)
        except Exception as e:
            st.warning(f'Grid table loaded, but plot could not be rendered: {e}')
        st.dataframe(efit_frame.head(12), use_container_width=True)
        st.download_button('Download EFIT grid CSV', data=efit_frame.to_csv(index=False).encode('utf-8'), file_name=f"mast_{efit_manifest['shot']}_efit_grid.csv", mime='text/csv')
        st.download_button('Download EFIT manifest', data=json.dumps(efit_manifest, indent=2), file_name='mast_efit_manifest.json', mime='application/json')
        st.caption(efit_manifest['scientific_limit'])

elif page=='Data Import & QC':
    st.title('Data Import & Quality Control')
    if df is None: st.info('Use the upload control in the left panel. Raw input is preserved and never overwritten.'); st.stop()
    st.success(f'{filename}: {len(df):,} rows × {len(df.columns)} columns')
    m1,m2,m3,m4=st.columns(4); m1.metric('Rows',f'{len(df):,}'); m2.metric('Columns',len(df.columns)); m3.metric('Numeric columns',len(numeric)); m4.metric('SHA-256',sha[:12]+'…')
    st.dataframe(df.head(50),use_container_width=True)
    st.markdown('### Quality audit')
    qcol=col_select('Signal / column',numeric,'qc_col')
    q=signal_quality(pd.to_numeric(df[qcol],errors='coerce').to_numpy()); st.json(q)
    missing=pd.DataFrame({'column':df.columns.astype(str),'dtype':[str(t) for t in df.dtypes],'missing_count':df.isna().sum().values,'missing_fraction':df.isna().mean().values})
    st.dataframe(missing,use_container_width=True)
    plot_line(np.arange(len(df)),pd.to_numeric(df[qcol],errors='coerce'),'Sample index',qcol,'Quality preview')
    json_download('Export QC report JSON',{'file':filename,'sha256':sha,'quality':q,'columns':missing.to_dict(orient='records')},'tis_qc_report.json')
    download_df('Export imported table CSV',df,'tis_imported_data.csv')

elif page=='Time Series & MHD':
    st.title('Time Series, Spectral & MHD Analysis'); need_data()
    tcol=st.selectbox('Time column (or sample index)',['<sample index>']+numeric,key='ts_t')
    scol=col_select('Signal channel',numeric,'ts_s')
    fs0=st.number_input('Sampling frequency if no time column (Hz; required for sample-index mode)',min_value=0.0,value=0.0,format='%.8g',help='No assumed 10 kHz default: enter the actual acquisition rate when no calibrated time column is available.')
    if tcol=='<sample index>' and fs0<=0:
        st.warning('A sampling rate is required because no time column is selected. Enter the acquisition rate above before interpreting frequency axes.')
        st.stop()
    gap_policy=st.selectbox('Irregular sampling / gap policy',['error','longest_segment','interpolate'],index=0,key='ts_gap_policy',
        format_func=lambda v:{'error':'Reject (strict)','longest_segment':'Keep longest gap-free segment (no interpolation)','interpolate':'Interpolate short gaps, keep longest finite run'}[v],
        help='Strict mode refuses gappy records. The other modes are explicit repairs and are recorded in the on-page report.')
    c1,c2,c3=st.columns(3)
    with c1: hp=st.number_input('High-pass cutoff (0=off)',min_value=0.0,value=0.0)
    with c2: lp=st.number_input('Low-pass cutoff (0=off)',min_value=0.0,value=0.0)
    with c3: order=st.slider('Butterworth order',1,8,3)
    try:
        t,y,fs,prep_report=cached_prepare_time_series(None if tcol=='<sample index>' else pd.to_numeric(df[tcol],errors='coerce').to_numpy(dtype=float),pd.to_numeric(df[scol],errors='coerce').to_numpy(dtype=float),fs0,gap_policy)
        yf=cached_filter(y,fs,hp or None,lp or None,order)
    except Exception as e:
        st.error(f'Time-series preparation failed: {e}')
        if tcol!='<sample index>':
            try:
                with st.expander('Sampling audit of the selected time column'):
                    st.json(sampling_audit(pd.to_numeric(df[tcol],errors='coerce').to_numpy(dtype=float)))
                st.info('If gaps are real (burst acquisition, dropped samples), choose a gap policy above instead of editing the data by hand.')
            except Exception: pass
        st.stop()
    if prep_report.get('modified'): st.warning('Data were repaired before analysis: '+prep_report.get('note','')+f" (policy: {prep_report['gap_policy']}).")
    if tcol!='<sample index>':
        with st.expander('Sampling audit'):
            st.json(sampling_audit(t))
    with st.expander('Optional: despike (Hampel) before analysis'):
        st.caption('Removes isolated glitches only. Real transients such as ELMs are removed too if the window is short, so this is OFF by default.')
        if st.checkbox('Apply Hampel despiking to the analysis channel',value=False,key='ts_despike'):
            dw=st.number_input('Window (odd samples)',min_value=3,max_value=2001,value=11,step=2,key='ts_despike_w'); ds=st.number_input('Threshold (robust σ)',min_value=1.0,value=6.0,key='ts_despike_s')
            try:
                dres=hampel_despike(y,int(dw),float(ds)); y=dres['values']; yf=cached_filter(y,fs,hp or None,lp or None,order)
                st.info(f"Replaced {dres['n_replaced']} sample(s). {dres['method']}")
            except Exception as e: st.error(str(e))

    x1,x2,x3=st.columns(3); x1.metric('Finite samples',len(y)); x2.metric('Sampling rate',f'{fs:.6g} Hz'); x3.metric('Nyquist',f'{fs/2:.6g} Hz')
    plot_line(t,y,'Time (s)',scol,'Raw signal'); plot_line(t,yf,'Time (s)',scol,'Filtered signal')
    st.markdown('### Welch PSD settings')
    wc1,wc2,wc3=st.columns(3)
    with wc1: welch_n=st.number_input('Segment length (samples)',min_value=8,max_value=int(len(y)),value=min(4096,int(len(y))),step=1,key='welch_nperseg')
    with wc2: welch_overlap=st.number_input('Overlap (samples)',min_value=0,max_value=max(0,int(welch_n)-1),value=int(welch_n)//2,step=1,key='welch_overlap')
    with wc3: welch_window=st.selectbox('Window',['hann','hamming','blackman','boxcar','bartlett'],index=0,key='welch_window')
    fft_window=st.selectbox('FFT amplitude window',['hann','flattop','hamming','blackman','boxcar'],index=0,key='fft_window')
    f,amp=cached_fft(yf,fs,fft_window)
    st.caption(f'FFT window: {fft_window}; amplitude scaling uses the selected window coherent gain. Flat-top is available for reduced off-bin amplitude bias.')
    fp,psd=cached_welch(yf,fs,int(welch_n),int(welch_overlap),welch_window)
    st.caption(f'Welch nominal frequency-bin spacing: {fs/int(welch_n):.6g} Hz; overlap: {int(welch_overlap)/int(welch_n):.0%}.')
    plot_line(f,amp,'Frequency (Hz)','Amplitude','FFT amplitude'); plot_line(fp,np.maximum(psd,np.finfo(float).tiny),'Frequency (Hz)','PSD (unit²/Hz)','Welch PSD',True)
    st.markdown('### Time-frequency spectrogram')
    tf_n=st.number_input('Spectrogram segment length',min_value=8,max_value=int(len(y)),value=min(256,int(len(y))),step=1,key='tf_n')
    tf_overlap=st.number_input('Segment overlap (samples)',min_value=0,max_value=int(tf_n)-1,value=int(tf_n)//2,step=1,key='tf_overlap')
    if st.button('Calculate spectrogram'):
        try:
            tf=time_frequency_spectrogram(yf,fs,nperseg=int(tf_n),noverlap=int(tf_overlap))
            fig=Figure(figsize=(9,4)); ax=fig.subplots()
            mesh=ax.pcolormesh(tf['time_s'],tf['frequency_Hz'],10*np.log10(np.maximum(tf['psd'],np.finfo(float).tiny)),shading='auto')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)'); ax.set_title('Short-time power spectral density (dB)')
            fig.colorbar(mesh,ax=ax,label='PSD (dB re 1 unit²/Hz)'); fig.tight_layout(); st.pyplot(fig)
            download_df('Download spectrogram matrix',pd.DataFrame(tf['psd'],index=tf['frequency_Hz'],columns=[f'{v:.9g}' for v in tf['time_s']]).reset_index(names='frequency_Hz'),'tis_spectrogram.csv')
        except Exception as e: st.error(str(e))
    st.markdown('### Autocorrelation / correlation-time estimate')
    max_lag=st.number_input('Maximum autocorrelation lag (s)',min_value=0.0,value=min(0.1,float(t[-1]-t[0])),max_value=float(t[-1]-t[0]),format='%.8g',key='acf_max_lag')
    if st.button('Analyze autocorrelation'):
        try:
            acf=autocorrelation_analysis(yf,fs,max_lag_s=max_lag)
            st.metric('First 1/e correlation time',f"{acf['e_folding_time_s']:.6g} s" if acf['e_folding_time_s'] is not None else 'Not reached in selected lag range')
            plot_line(acf['lag_s'],acf['autocorrelation'],'Lag (s)','Normalized autocorrelation','Autocorrelation')
            download_df('Download autocorrelation',pd.DataFrame({'lag_s':acf['lag_s'],'autocorrelation':acf['autocorrelation']}),'tis_autocorrelation.csv')
        except Exception as e: st.error(str(e))
    download_df('Download filtered series',pd.DataFrame({'time_s':t,'raw':y,'filtered':yf}),'tis_timeseries.csv')
    download_df('Download FFT',pd.DataFrame({'frequency_Hz':f,'amplitude':amp}),'tis_fft.csv'); download_df('Download PSD',pd.DataFrame({'frequency_Hz':fp,'psd':psd}),'tis_welch_psd.csv')
    st.markdown('### Cross-channel coherence / phase / delay')
    others=[c for c in numeric if c!=scol]
    if others:
        other=st.selectbox('Second synchronized channel',others,key='cross_other')
        # Apply one shared validity mask before any NaN removal so samples remain paired.
        raw_x=pd.to_numeric(df[scol],errors='coerce').to_numpy(dtype=float)
        raw_y=pd.to_numeric(df[other],errors='coerce').to_numpy(dtype=float)
        raw_t=None if tcol=='<sample index>' else pd.to_numeric(df[tcol],errors='coerce').to_numpy(dtype=float)
        common=np.isfinite(raw_x)&np.isfinite(raw_y)
        if raw_t is not None: common &= np.isfinite(raw_t)
        tx=None if raw_t is None else raw_t[common]
        try:
            _,x_pair,fs_pair=prepare_time_series(tx,raw_x[common],fs0)
            _,y_pair,fs2=prepare_time_series(tx,raw_y[common],fs0)
            if len(x_pair)==len(y_pair) and np.isclose(fs_pair,fs2) and np.isclose(fs,fs_pair):
                x_pair_f=butterworth_filter(x_pair,fs_pair,hp or None,lp or None,order)
                y2f=butterworth_filter(y_pair,fs2,hp or None,lp or None,order)
                xa=cross_analysis(x_pair_f,y2f,fs_pair)
                lags,corr,lag=cross_correlation_lag(x_pair_f,y2f,fs_pair)
                st.caption('Both channels use the same timestamp-validity mask and the same filter settings. Lag convention: a y-channel delayed relative to x gives a negative lag.')
                st.metric('Peak absolute-correlation lag',f'{lag:.6g} s')
                plot_line(xa['frequency_Hz'],xa['coherence'],'Frequency (Hz)','Coherence','Magnitude-squared coherence')
                plot_line(xa['frequency_Hz'],xa['coherence_95pct_null_threshold'],'Frequency (Hz)','Coherence threshold','Approximate 95% null threshold')
                plot_line(xa['frequency_Hz'],xa['cross_phase_rad'],'Frequency (Hz)','Phase (rad)','Cross phase φy−φx')
                plot_line(xa['frequency_Hz'],xa['phase_standard_error_approx_rad'],'Frequency (Hz)','Phase SE (rad)','Approximate cross-phase standard error')
                st.caption('Uncertainty curves use nominal Welch segment count; 50% overlap reduces independent averages, so treat them as diagnostics rather than exact confidence intervals.')
                plot_line(lags,corr,'Lag (s)','Correlation','Cross-correlation')
                download_df('Download cross-channel results',pd.DataFrame(xa),'tis_cross_analysis.csv')
            else: st.warning('Channels need aligned timestamps and compatible sampling rates.')
        except Exception as e: st.warning(f'Cross-channel analysis unavailable: {e}')
    st.markdown('### Mode number from synchronized sensor channels')
    st.caption('Select at least three synchronized channels in sensor order and enter their toroidal/poloidal angles. Relative phases are extracted from cross spectra and weighted by coherence.')
    mode_channels=st.multiselect('Sensor signal channels',numeric,default=[],key='mode_sensor_channels')
    mode_angles=st.text_input('Sensor angles θ (radians, same order as channels)','0,0.785398,1.570796,2.356194,3.141593,3.926991,4.712389,5.497787',key='mode_sensor_angles')
    mode_coh=st.slider('Minimum coherence with reference sensor',0.0,1.0,0.2,0.05,key='mode_min_coherence')
    mode_conv=st.selectbox('Mode-number sign convention',['wave_cos_wt_minus_n_theta','phase_equals_n_theta'],index=0,key='mode_convention',
        help='wave_cos_wt_minus_n_theta: n>0 for a wave cos(ωt−nθ) travelling toward +θ (the usual convention). phase_equals_n_theta: raw fit of cross-phase = +nθ (opposite sign).')
    if st.button('Extract phases and fit mode number'):
        try:
            if len(mode_channels)<3: raise ValueError('Select at least three sensor channels.')
            angles=np.asarray([float(z.strip()) for z in mode_angles.split(',')],dtype=float)
            if len(angles)!=len(mode_channels): raise ValueError('The number of sensor angles must match the selected channels.')
            raw_signals=np.column_stack([pd.to_numeric(df[name],errors='coerce').to_numpy(dtype=float) for name in mode_channels])
            raw_times=None if tcol=='<sample index>' else pd.to_numeric(df[tcol],errors='coerce').to_numpy(dtype=float)
            valid=np.all(np.isfinite(raw_signals),axis=1)
            if raw_times is not None: valid &= np.isfinite(raw_times)
            t_common=None if raw_times is None else raw_times[valid]
            clean=[]
            fs_common=None
            for j in range(raw_signals.shape[1]):
                _,yj,fsj=prepare_time_series(t_common,raw_signals[valid,j],fs0)
                if fs_common is not None and not np.isclose(fsj,fs_common): raise ValueError('Sensor channels do not share a common sampling rate.')
                fs_common=fsj; clean.append(yj)
            result=mode_number_from_channels(angles,np.vstack(clean),fs_common,min_coherence=mode_coh,convention=mode_conv)
            st.json(result)
            json_download('Export channel-derived mode fit',result,'tis_mode_number_from_channels.json')
        except Exception as e: st.error(str(e))
    with st.expander('Advanced: fit already-extracted phases'):
        ang=st.text_input('Sensor angles θ (radians, comma-separated)','0,1.5707963,3.1415927,4.712389',key='manual_mode_angles')
        phs=st.text_input('Measured phase (radians, comma-separated)','0,1.5707963,3.1415927,4.712389',key='manual_mode_phases')
        if st.button('Fit supplied phases',key='fit_manual_mode'):
            try: st.json(mode_number_from_phase([float(z) for z in ang.split(',')],[float(z) for z in phs.split(',')],convention=mode_conv))
            except Exception as e: st.error(str(e))
    st.markdown('### Calibrated signal integration')
    st.caption('Useful for derivative-type diagnostics such as pickup coils or Rogowski signals only when the sensor scale, offset/baseline, bandwidth and initial condition are known. The software does not infer these calibrations.')
    int_scale=st.number_input('Integration input scale',value=1.0,format='%.8g',key='int_scale')
    int_initial=st.number_input('Integrated initial value',value=0.0,format='%.8g',key='int_initial')
    use_baseline=st.checkbox('Subtract mean over a baseline interval',value=False,key='int_use_baseline')
    if use_baseline:
        tmin=float(np.min(t)); tmax=float(np.max(t))
        b1,b2=st.columns(2)
        with b1: baseline_start=st.number_input('Baseline start (s)',value=tmin, min_value=tmin, max_value=tmax, key='int_baseline_start')
        with b2: baseline_end=st.number_input('Baseline end (s)',value=min(tmin+(tmax-tmin)*0.1,tmax), min_value=tmin, max_value=tmax, key='int_baseline_end')
    else:
        baseline_start=baseline_end=None
    drift_start=drift_end=None
    if use_baseline and st.checkbox('Also remove linear drift using a post-event quiet window',value=False,key='int_use_drift'):
        d1,d2=st.columns(2)
        with d1: drift_start=st.number_input('Post window start (s)',value=max(tmin,tmax-(tmax-tmin)*0.1),min_value=tmin,max_value=tmax,key='int_drift_start')
        with d2: drift_end=st.number_input('Post window end (s)',value=tmax,min_value=tmin,max_value=tmax,key='int_drift_end')
        st.caption('Assumes the true signal is ≈0 in both windows, so any difference of their means is instrument drift (linear in time).')
    if st.button('Integrate calibrated signal'):
        try:
            integrated=integrate_time_series(t,y,scale=int_scale,initial_value=int_initial,baseline_start_s=baseline_start,baseline_end_s=baseline_end,drift_start_s=drift_start,drift_end_s=drift_end)
            st.json({k:v for k,v in integrated.items() if k not in {'time_s','calibrated_signal','integrated_signal'}})
            plot_line(t,integrated['integrated_signal'],'Time (s)','Integrated signal','Cumulative calibrated integral')
            out=pd.DataFrame({'time_s':t,'input_signal':y,'calibrated_signal':integrated['calibrated_signal'],'integrated_signal':integrated['integrated_signal']})
            download_df('Download integrated series',out,'tis_integrated_timeseries.csv')
            json_download('Export integration manifest',{k:v for k,v in integrated.items() if k not in {'time_s','calibrated_signal','integrated_signal'}},'tis_integration_manifest.json')
        except Exception as e: st.error(str(e))

elif page=='Events & Transients':
    st.title('Events & Transients: ELMs, Sawteeth, Current Quench'); need_data()
    st.caption('Statistical detectors with robust (MAD) thresholds. Always confirm detections against the raw trace; ELM type, pellets and disruptions are not classified.')
    kind=st.radio('Analysis',['ELMs (Dα / divertor emission)','Sawteeth (core SXR / ECE)','Current quench (Iₚ)'],horizontal=True,key='ev_kind')
    ev_t=col_select('Time column (s)',numeric,'ev_t'); ev_y=col_select('Signal',[c for c in numeric if c!=ev_t] or numeric,'ev_y')
    tv=pd.to_numeric(df[ev_t],errors='coerce').to_numpy(dtype=float); yv=pd.to_numeric(df[ev_y],errors='coerce').to_numpy(dtype=float)
    fin=np.isfinite(tv)&np.isfinite(yv); tv,yv=tv[fin],yv[fin]
    if len(tv)<32: st.error('Too few finite samples.'); st.stop()
    t_lo,t_hi=float(tv.min()),float(tv.max())
    win=st.slider('Analysis window (s)',t_lo,t_hi,(t_lo,t_hi),key='ev_window') if t_hi>t_lo else (t_lo,t_hi)
    m=(tv>=win[0])&(tv<=win[1]); tw,yw=tv[m],yv[m]
    if kind.startswith('ELMs'):
        nsig=st.number_input('Detection threshold (robust σ above baseline)',min_value=1.0,value=5.0,key='ev_elm_sigma')
        if st.button('Detect ELMs',type='primary'):
            try:
                r=detect_elms(tw,yw,n_sigmas=float(nsig))
                fig=Figure(figsize=(9,3.3)); ax=fig.subplots(); ax.plot(tw,yw,lw=.7); ax.plot(r['event_times_s'],np.interp(r['event_times_s'],tw,yw),'rv',ms=4,label=f"{r['n_elms']} ELMs"); ax.set_xlabel('Time (s)'); ax.set_ylabel(ev_y); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig)
                summary={k:v for k,v in r.items() if k not in {'event_times_s','amplitudes_above_baseline','instantaneous_frequency_Hz','instantaneous_frequency_time_s'}}
                st.json(summary)
                out=pd.DataFrame({'elm_time_s':r['event_times_s'],'amplitude_above_baseline':r['amplitudes_above_baseline']}); download_df('Download ELM list',out,'tis_elm_events.csv')
                json_download('Export ELM analysis',summary,'tis_elm_analysis.json'); add_report_section('ELM detection',summary,fig)
            except Exception as e: st.error(str(e))
    elif kind.startswith('Sawteeth'):
        nsig=st.number_input('Detection threshold (robust σ of −dS/dt)',min_value=1.0,value=5.0,key='ev_saw_sigma')
        if st.button('Detect sawtooth crashes',type='primary'):
            try:
                r=detect_sawteeth(tw,yw,n_sigmas=float(nsig))
                fig=Figure(figsize=(9,3.3)); ax=fig.subplots(); ax.plot(tw,yw,lw=.7); ax.plot(r['crash_times_s'],np.interp(r['crash_times_s'],tw,yw),'rv',ms=5,label=f"{r['n_crashes']} crashes"); ax.set_xlabel('Time (s)'); ax.set_ylabel(ev_y); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig)
                summary={k:v for k,v in r.items() if k not in {'crash_times_s','crash_amplitudes','max_drop_rates_per_s'}}
                st.json(summary)
                out=pd.DataFrame({'crash_time_s':r['crash_times_s'],'crash_amplitude':r['crash_amplitudes'],'max_drop_rate_per_s':r['max_drop_rates_per_s']}); download_df('Download crash list',out,'tis_sawtooth_events.csv')
                json_download('Export sawtooth analysis',summary,'tis_sawtooth_analysis.json'); add_report_section('Sawtooth analysis',summary,fig)
            except Exception as e: st.error(str(e))
    else:
        i0=st.number_input('Reference (flat-top) current, A (0 = 95th percentile of |Iₚ|)',min_value=0.0,value=0.0,format='%.8g',key='ev_cq_i0')
        if st.button('Analyze current quench',type='primary'):
            try:
                r=current_quench_analysis(tw,yw,reference_current_A=i0 or None)
                fig=Figure(figsize=(9,3.3)); ax=fig.subplots(); ax.plot(tw,yw,lw=.8)
                for k in ('t_upper_s','t_lower_s'):
                    if r.get(k) is not None: ax.axvline(r[k],color='r',ls='--',lw=.8)
                ax.set_xlabel('Time (s)'); ax.set_ylabel(ev_y); ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig)
                st.json(r); json_download('Export current-quench analysis',r,'tis_current_quench.json'); add_report_section('Current quench',r,fig)
            except Exception as e: st.error(str(e))

elif page=='Confinement & Scalings':
    st.title('Confinement, Scalings & Dimensionless Parameters')
    st.caption('Textbook and empirical relations (IPB98(y,2), Martin-08 L–H threshold, ITER q95 formula, Troyon β_N, collisionality). Use volume/line averages; these are comparison tools, not device-validated predictions.')
    tabs=st.tabs(['Confinement summary','β, collisionality, resistivity'])
    with tabs[0]:
        a,b,c=st.columns(3)
        with a:
            cs_ip=st.number_input('Plasma current Iₚ (A)',min_value=1.0,value=1.0e6,format='%.6g',key='cs_ip'); cs_bt=st.number_input('Toroidal field Bₜ (T)',min_value=0.01,value=1.0,key='cs_bt'); cs_mass=st.number_input('Main ion mass (amu)',min_value=1.0,value=2.0,key='cs_mass')
        with b:
            cs_R=st.number_input('Major radius R (m)',min_value=0.01,value=0.9,key='cs_R'); cs_a=st.number_input('Minor radius a (m)',min_value=0.005,value=0.55,key='cs_a'); cs_k=st.number_input('Elongation κ',min_value=1.0,value=1.8,key='cs_k')
        with c:
            cs_ne=st.number_input('Line-averaged nₑ (m⁻³)',min_value=1e15,value=4e19,format='%.6g',key='cs_ne'); cs_p=st.number_input('Net loss power P_loss (W)',min_value=1.0,value=2.0e6,format='%.6g',key='cs_p'); cs_W=st.number_input('Thermal stored energy W (J; 0 = omit)',min_value=0.0,value=0.0,format='%.6g',key='cs_W')
        cs_d=st.number_input('Triangularity δ',min_value=-1.0,max_value=1.0,value=0.3,key='cs_d')
        if st.button('Compute confinement summary',type='primary'):
            try:
                res=confinement_summary(ip_A=cs_ip,bt_T=cs_bt,r_m=cs_R,a_m=cs_a,kappa=cs_k,ne_line_avg_m3=cs_ne,p_loss_W=cs_p,mass_amu=cs_mass,stored_energy_J=cs_W or None,delta=cs_d)
                st.json(res); json_download('Export confinement summary',res,'tis_confinement_summary.json'); add_report_section('Confinement summary',res)
            except Exception as e: st.error(str(e))
    with tabs[1]:
        a,b=st.columns(2)
        with a:
            pb_ne=st.number_input('nₑ (m⁻³)',min_value=1e10,value=5e19,format='%.6g',key='pb_ne'); pb_te=st.number_input('Tₑ (eV)',min_value=0.1,value=1000.0,key='pb_te'); pb_ti=st.number_input('Tᵢ (eV)',min_value=0.1,value=800.0,key='pb_ti'); pb_z=st.number_input('Z_eff',min_value=1.0,value=1.5,key='pb_z')
        with b:
            pb_ip=st.number_input('Iₚ (A)',min_value=1.0,value=1.0e6,format='%.6g',key='pb_ip'); pb_bt=st.number_input('Bₜ (T)',min_value=0.01,value=1.0,key='pb_bt'); pb_R=st.number_input('R (m)',min_value=0.01,value=0.9,key='pb_R'); pb_a=st.number_input('a (m)',min_value=0.005,value=0.55,key='pb_a'); pb_k=st.number_input('κ',min_value=1.0,value=1.8,key='pb_k'); pb_q=st.number_input('Safety factor q (for ν*)',min_value=0.1,value=3.0,key='pb_q')
        if st.button('Compute dimensionless parameters',type='primary'):
            try:
                res={'beta':beta_metrics(pb_ne,pb_te,pb_ti,pb_ip,pb_bt,pb_a,pb_k),
                     'coulomb_logarithm':coulomb_logarithm_electron(pb_ne,pb_te),
                     'nu_star_electron':collisionality_nu_star(pb_ne,pb_te,pb_q,pb_R,pb_a),
                     'spitzer':spitzer_resistivity(pb_te,pb_z,ne_m3=pb_ne),
                     'gyro_radii_and_speeds':larmor_radii_and_speeds(pb_te,pb_ti,pb_bt,pb_ne)}
                st.json(res); json_download('Export dimensionless parameters',res,'tis_dimensionless_parameters.json'); add_report_section('Dimensionless parameters',res)
            except Exception as e: st.error(str(e))

elif page=='Spectral Analysis':
    st.title('Spectroscopy, X-ray & Energy Spectra'); need_data()
    st.caption('Input columns must be calibrated to the physical coordinate used. Background, instrument function and detector response are not inferred automatically.')
    mode=st.radio('Workflow',['Optical / emission line','CXRS line profile','Thomson spectral distribution','Molecular spectroscopy','Molecular Boltzmann temperature','Wavelength calibration','Line-ratio analysis','X-ray energy spectrum','Generic energy-spectrum moments'],horizontal=True)
    if mode=='Thomson spectral distribution':
        st.caption('Model-independent spectral-shape characterization. This workflow does not infer electron temperature/density or uniquely reconstruct an electron velocity distribution.')
        xcol=col_select('Calibrated Thomson wavelength (nm)',numeric,'tsd_x')
        ycol=col_select('Thomson spectral signal',numeric,'tsd_y')
        sigma_col=st.selectbox('Signal uncertainty σ (optional)', ['<none>']+numeric, key='tsd_sigma')
        inst_sigma=st.number_input('Gaussian instrument standard deviation σ (nm; independently calibrated)',min_value=1e-12,value=0.05,format='%.8g',key='tsd_instrument_sigma')
        background_fraction=st.slider('Fraction of each spectral edge used for background estimation',0.02,0.40,0.10,0.01,key='tsd_background_fraction')
        if st.button('Analyze Thomson spectral distribution'):
            try:
                _x=pd.to_numeric(df[xcol],errors='coerce').to_numpy(dtype=float)
                _y=pd.to_numeric(df[ycol],errors='coerce').to_numpy(dtype=float)
                _unc=None if sigma_col=='<none>' else pd.to_numeric(df[sigma_col],errors='coerce').to_numpy(dtype=float)
                _result=analyze_thomson_spectral_distribution(
                    _x,_y,instrument_sigma_nm=float(inst_sigma),
                    signal_uncertainty=_unc,background_fraction=float(background_fraction))
                st.session_state['tis_thomson_distribution']={
                    'xcol':xcol,'ycol':ycol,'sigma_col':sigma_col,'instrument_sigma_nm':float(inst_sigma),
                    'background_fraction':float(background_fraction),'data_sha':sha,'result':_result}
            except Exception as e:
                st.error(str(e))
        _saved_tsd=st.session_state.get('tis_thomson_distribution')
        if _saved_tsd and _saved_tsd.get('xcol')==xcol and _saved_tsd.get('ycol')==ycol and _saved_tsd.get('sigma_col')==sigma_col and _saved_tsd.get('instrument_sigma_nm')==float(inst_sigma) and _saved_tsd.get('background_fraction')==float(background_fraction) and _saved_tsd.get('data_sha')==sha:
            _r=_saved_tsd['result']
            _m1,_m2,_m3,_m4=st.columns(4)
            _m1.metric('Spectral centroid',f"{_r['spectral_centroid_nm']:.6g} nm")
            _m2.metric('Observed RMS width',f"{_r['observed_sigma_nm']:.6g} nm")
            _m3.metric('Instrument-corrected width',f"{_r['instrument_corrected_sigma_nm']:.6g} nm" if _r['instrument_width_resolved'] else "Unresolved")
            _m4.metric('Positive integrated area',f"{_r['integrated_positive_spectral_signal']:.6g}")
            if _r['instrument_width_resolved']:
                st.success('Instrument-corrected spectral width is resolved under the stated Gaussian instrument-response assumption.')
            else:
                st.warning('The observed spectral variance does not exceed the calibrated instrument variance. Instrument-corrected width is unresolved and is intentionally not reported.')
            st.caption('Descriptive spectral moments and Gaussian shape-fit diagnostics only. This workflow does not infer electron temperature, electron density, or a unique electron velocity distribution.')
            st.json({k:v for k,v in _r.items() if not isinstance(v,(np.ndarray,dict)) and k not in {'signal','wavelength_nm','background','background_subtracted_signal','positive_spectral_weights','fitted_spectrum','residual'}} | {'gaussian_fit_parameters':_r['gaussian_fit_parameters'],'gaussian_fit_parameter_uncertainties':_r['gaussian_fit_parameter_uncertainties']})
            st.line_chart(pd.DataFrame({'measured':_r['signal'],'background-subtracted':_r['background_subtracted_signal'],'Gaussian fit':_r['fitted_spectrum']} ,index=_r['wavelength_nm']))
            json_download('Export Thomson spectral-distribution results', {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in _r.items()}, 'thomson_spectral_distribution.json')
    elif mode=='Optical / emission line':
        xcol=col_select('Wavelength / spectral coordinate',numeric,'sp_x'); ycol=col_select('Intensity',numeric,'sp_y')
        model=st.selectbox('Line model',['Gaussian','Lorentzian','Voigt']); sigma_col=st.selectbox('Intensity uncertainty σ (optional)',['<none>']+numeric,key='sp_sigma'); xmin=st.number_input('Fit minimum',value=float(pd.to_numeric(df[xcol],errors='coerce').min())); xmax=st.number_input('Fit maximum',value=float(pd.to_numeric(df[xcol],errors='coerce').max()))
        if st.button('Fit spectral line'):
            try:
                fit_sigma=None if sigma_col=='<none>' else pd.to_numeric(df[sigma_col],errors='coerce')
                st.session_state['tis_spectral_fit']={'xcol':xcol,'ycol':ycol,'model':model,'sigma_col':sigma_col,'xmin':xmin,'xmax':xmax,'data_sha':sha,'fit':fit_spectral_line(df[xcol],df[ycol],xmin,xmax,model,sigma_y=fit_sigma)}
            except Exception as e: st.error(str(e))
        saved=st.session_state.get('tis_spectral_fit')
        if saved and saved.get('xcol')==xcol and saved.get('ycol')==ycol and saved.get('model')==model and saved.get('sigma_col')==sigma_col and saved.get('xmin')==xmin and saved.get('xmax')==xmax and saved.get('data_sha')==sha:
            fit=saved['fit']; st.json(fit)
            if model=='Gaussian':
                lam0=st.number_input('Known rest wavelength λ₀ (same unit as x)',min_value=1e-12,value=None,placeholder='Enter reference wavelength',key='doppler_rest_wavelength')
                sigma_lam0=st.number_input('Rest-wavelength uncertainty',min_value=0.0,value=0.0,format='%.8g')
                mass=st.number_input('Ion mass (amu; enter the actual emitting species)',min_value=0.0,value=0.0,key='doppler_mass_amu')
                sigma_mass=st.number_input('Ion-mass uncertainty (amu)',min_value=0.0,value=0.0,format='%.8g')
                inst=st.number_input('Instrument FWHM (same x unit; from independent calibration)',min_value=0.0,value=0.0,format='%.8g')
                assume_inst_negligible=st.checkbox('Assume instrumental broadening is negligible (must be independently justified)',value=False,key='doppler_inst_negligible')
                sigma_inst=st.number_input('Instrument FWHM uncertainty',min_value=0.0,value=0.0,format='%.8g')
                if st.button('Estimate Doppler quantities'):
                    if lam0 is None:
                        st.error('Enter the independently known rest wavelength; using the fitted center would force the velocity to zero.')
                    elif mass<=0:
                        st.error('Enter the actual emitting-ion mass. No isotope is assumed by default.')
                    elif inst==0 and not assume_inst_negligible:
                        st.error('Enter the calibrated instrument width, or explicitly justify the negligible-instrument-width assumption.')
                    else:
                        # Do not infer a thermal width from an under-sampled line.
                        # A Gaussian optimizer can return sub-pixel widths even when
                        # the detector sampling does not resolve the physical profile.
                        _sx=np.asarray(pd.to_numeric(df[xcol],errors='coerce'),dtype=float)
                        _sx=np.sort(_sx[np.isfinite(_sx)])
                        _steps=np.diff(_sx)
                        _steps=_steps[np.isfinite(_steps)&(_steps>0)]
                        _median_step=float(np.median(_steps)) if len(_steps) else float('nan')
                        _observed_fwhm=float(fit.get('FWHM',float('nan')))
                        if not np.isfinite(_median_step) or _observed_fwhm < 3.0*_median_step:
                            st.error(
                                f"Temperature withheld: the fitted FWHM ({_observed_fwhm:.6g} in the x-axis unit) "
                                f"is not resolved by the sampling (median spacing {_median_step:.6g}; "
                                "require at least 3 samples across FWHM for this workflow). "
                                "Use a genuinely wavelength-calibrated, higher-resolution spectrum and "
                                "independently measured instrument line-spread function."
                            )
                        else:
                            try:
                                thermal=doppler_temperature_from_fwhm_eV(
                                    observed_fwhm=float(fit['FWHM']),
                                    lambda0=float(lam0),
                                    mass_amu=float(mass),
                                    instrumental_fwhm=float(inst),
                                    sigma_observed_fwhm=float(fit.get('standard_error_sigma',0.0))*np.sqrt(8.0*np.log(2.0)),
                                    sigma_instrumental_fwhm=float(sigma_inst),
                                )
                                velocity=doppler_velocity_with_uncertainty_m_s(float(fit['center']),lam0,float(fit.get('standard_error_center',0.0)),sigma_lam0)
                                st.json({'doppler_temperature_eV':thermal['temperature_eV'],'doppler_temperature_standard_uncertainty_eV':thermal['standard_uncertainty_eV'],'relative_temperature_standard_uncertainty':thermal['relative_standard_uncertainty'],'observed_fwhm':thermal['observed_fwhm'],'instrumental_fwhm':thermal['instrumental_fwhm'],'corrected_doppler_fwhm':thermal['doppler_fwhm'],'line_of_sight_velocity_m_s':velocity['line_of_sight_velocity_m_s'],'line_of_sight_velocity_standard_uncertainty_m_s':velocity['standard_uncertainty_m_s'],'temperature_uncertainty_method':'First-order propagation of independent observed/instrument FWHM uncertainties; excludes systematic uncertainty.','assumptions':thermal['warning']})
                            except Exception as e:
                                st.error(f'Ion temperature withheld: {e}')
            plot_line(np.asarray(df[xcol],float),np.asarray(df[ycol],float),xcol,ycol,'Input spectrum')
            json_download('Export fit JSON',fit,'tis_spectral_fit.json')
    elif mode=='CXRS line profile':
        st.caption('Fit one calibrated, isolated charge-exchange line. Requires independently known rest wavelength, emitting-ion mass and Gaussian instrumental sigma (not FWHM). Reports line-of-sight velocity and a single-Maxwellian Ti estimate; it does not infer impurity density or invert multiple views.')
        xcol=col_select('Calibrated wavelength (nm)',numeric,'cxrs_x')
        ycol=col_select('Spectral intensity',numeric,'cxrs_y')
        c1,c2=st.columns(2)
        with c1:
            rest=st.number_input('Known rest wavelength λ₀ (nm)',min_value=0.000001,value=530.0,format='%.9g',key='cxrs_rest')
            mass=st.number_input('Emitting-ion mass (amu)',min_value=0.000001,value=12.0,format='%.8g',key='cxrs_mass')
        with c2:
            inst=st.number_input('Instrument Gaussian sigma (nm; not FWHM)',min_value=0.000001,value=0.008,format='%.9g',key='cxrs_inst_sigma')
            inst_unc=st.number_input('Instrument sigma uncertainty (nm; 1σ)',min_value=0.0,value=0.0,format='%.9g',key='cxrs_inst_sigma_unc')
            viewing_angle=st.number_input('Viewing angle metadata (rad; no deprojection)',min_value=0.0,max_value=float(np.pi),value=0.0,format='%.8g',key='cxrs_viewing_angle')
            sigma_col=st.selectbox('Per-sample intensity uncertainty σ (optional)',['<none>']+numeric,key='cxrs_sigma')
        if st.button('Analyze CXRS line'):
            try:
                _spec=pd.DataFrame({'wavelength_nm':pd.to_numeric(df[xcol],errors='coerce'),'signal':pd.to_numeric(df[ycol],errors='coerce')}).dropna()
                if sigma_col!='<none>':
                    _spec['signal_uncertainty']=pd.to_numeric(df.loc[_spec.index,sigma_col],errors='coerce')
                    if not np.all(np.isfinite(_spec['signal_uncertainty'])) or np.any(_spec['signal_uncertainty']<=0):
                        raise ValueError('Selected uncertainty column must contain finite positive values for every retained sample.')
                _spec=_spec.sort_values('wavelength_nm',kind='stable')
                _kwargs={}
                if sigma_col!='<none>': _kwargs['signal_uncertainty']=_spec['signal_uncertainty'].to_numpy(dtype=float)
                _result=analyze_cxrs_line(_spec['wavelength_nm'].to_numpy(dtype=float),_spec['signal'].to_numpy(dtype=float),
                    rest_wavelength_nm=float(rest),ion_mass_amu=float(mass),instrument_sigma_nm=float(inst),
                    instrument_sigma_uncertainty_nm=float(inst_unc),viewing_angle_rad=float(viewing_angle),**_kwargs)
                _result_dict=_result.to_dict()
                _result_dict['wavelength_nm']=_spec['wavelength_nm'].to_numpy(dtype=float).tolist()
                _result_dict['measured_signal']=_spec['signal'].to_numpy(dtype=float).tolist()
                st.session_state['tis_cxrs_result']={'data_sha':sha,'xcol':xcol,'ycol':ycol,'rest':float(rest),'mass':float(mass),'inst':float(inst),'inst_unc':float(inst_unc),'viewing_angle':float(viewing_angle),'sigma_col':sigma_col,'result':_result_dict}
            except Exception as e:
                st.error(str(e))
        _saved_cxrs=st.session_state.get('tis_cxrs_result')
        if _saved_cxrs and all(_saved_cxrs.get(k)==v for k,v in {'data_sha':sha,'xcol':xcol,'ycol':ycol,'rest':float(rest),'mass':float(mass),'inst':float(inst),'inst_unc':float(inst_unc),'viewing_angle':float(viewing_angle),'sigma_col':sigma_col}.items()):
            _rr=_saved_cxrs['result']
            st.json({k:v for k,v in _rr.items() if k not in {'fitted_signal','residual'}})
            st.line_chart(pd.DataFrame({'measured':_rr['measured_signal'],'fitted':_rr['fitted_signal'],'residual':_rr['residual']},index=_rr['wavelength_nm']))
            json_download('Export CXRS line-profile result',_rr,'tis_cxrs_line_profile.json')
            st.caption('Ti uncertainty includes local fit covariance and the supplied instrumental-sigma uncertainty; omitted instrument uncertainty defaults to zero. Other calibration/systematic effects remain excluded. Check the stated assumptions before interpreting Ti or velocity.')
    elif mode=='Molecular spectroscopy':
        st.caption('Exploratory molecular-band feature extraction. Wavelength calibration is required; no species or rotational/vibrational temperature is inferred automatically.')
        xcol=col_select('Calibrated wavelength (nm)',numeric,'mol_x')
        ycol=col_select('Molecular spectral intensity',numeric,'mol_y')
        prominence=st.slider('Peak prominence / spectral contrast',0.01,1.0,0.03,0.01,key='mol_prominence')
        baseline_q=st.slider('Constant baseline quantile',0.0,0.49,0.10,0.01,key='mol_baseline_quantile')
        refs_text=st.text_input('Optional user reference bandheads (nm:label; nm:label)',key='mol_refs')
        ranges_text=st.text_input('Optional integration windows (lower_nm:upper_nm; ...)',key='mol_ranges')
        if st.button('Analyze molecular spectrum'):
            try:
                refs=None
                if refs_text.strip():
                    refs=[]
                    for item in refs_text.split(';'):
                        parts=item.strip().split(':',1)
                        if len(parts)!=2 or not parts[1].strip():
                            raise ValueError('Reference entries must use wavelength_nm:label, separated by semicolons.')
                        refs.append((float(parts[0].strip()),parts[1].strip()))
                ranges=None
                if ranges_text.strip():
                    ranges=[]
                    for item in ranges_text.split(';'):
                        parts=item.strip().split(':',1)
                        if len(parts)!=2:
                            raise ValueError('Integration windows must use lower_nm:upper_nm, separated by semicolons.')
                        ranges.append((float(parts[0].strip()),float(parts[1].strip())))
                spectrum=pd.DataFrame({
                    'wavelength_nm':pd.to_numeric(df[xcol],errors='coerce'),
                    'intensity':pd.to_numeric(df[ycol],errors='coerce')
                }).dropna()
                spectrum=spectrum.sort_values('wavelength_nm',kind='stable')
                result=analyze_molecular_spectrum(
                    spectrum['wavelength_nm'].to_numpy(dtype=float),
                    spectrum['intensity'].to_numpy(dtype=float),
                    reference_bandheads=refs,
                    prominence_fraction=prominence,
                    baseline_quantile=baseline_q,
                    band_ranges_nm=ranges,
                )
                st.session_state['tis_molecular_spectrum_result']={
                    'data_sha':sha,'xcol':xcol,'ycol':ycol,'result':result
                }
            except Exception as e:
                st.error(str(e))
        saved_molecular=st.session_state.get('tis_molecular_spectrum_result')
        if saved_molecular and saved_molecular.get('data_sha')==sha and saved_molecular.get('xcol')==xcol and saved_molecular.get('ycol')==ycol:
            st.json(saved_molecular['result'])
            json_download('Export molecular spectroscopy JSON',saved_molecular['result'],'tis_molecular_spectroscopy.json')
            st.caption('Reference matches are candidates only. Integrated signal is intensity·nm, not radiated power unless the input is absolutely calibrated.')
        st.markdown('#### Optional empirical Gaussian-band fit')
        st.caption('Fits a sum of Gaussian features plus a constant baseline. This is a phenomenological fit, not a molecular transition/population model and does not infer rotational or vibrational temperature.')
        centers_text=st.text_input('Initial band-center guesses (nm, comma-separated)',key='mol_fit_centers')
        if st.button('Fit empirical Gaussian bands',key='mol_fit_button'):
            try:
                if not centers_text.strip():
                    raise ValueError('Enter one or more initial band-center guesses in nm.')
                fit_data=pd.DataFrame({
                    'wavelength_nm':pd.to_numeric(df[xcol],errors='coerce'),
                    'intensity':pd.to_numeric(df[ycol],errors='coerce')
                }).dropna().sort_values('wavelength_nm',kind='stable')
                wx=fit_data['wavelength_nm'].to_numpy(dtype=float)
                wy=fit_data['intensity'].to_numpy(dtype=float)
                if wx.size < 5 or not np.all(np.isfinite(wx)) or not np.all(np.isfinite(wy)):
                    raise ValueError('At least five finite wavelength/intensity samples are required.')
                if np.any(np.diff(wx) <= 0):
                    raise ValueError('Wavelength samples must be unique; remove duplicate wavelengths before fitting.')
                centers=[float(v.strip()) for v in centers_text.split(',') if v.strip()]
                if not centers or not np.all(np.isfinite(centers)) or any(c < wx[0] or c > wx[-1] for c in centers):
                    raise ValueError('All initial band centers must be finite and inside the measured wavelength range.')
                npeak=len(centers)
                if wx.size <= 3*npeak+1:
                    raise ValueError('Use fewer Gaussian components: the spectrum must have more samples than fitted parameters.')
                span=float(wx[-1]-wx[0])
                dx=float(np.min(np.diff(wx)))
                baseline0=float(np.quantile(wy,0.1))
                amp0=max(float(np.max(wy)-baseline0),np.finfo(float).eps)/npeak
                width0=max(span/(10*npeak),dx)
                names=[]
                p0=[]
                lower=[]
                upper=[]
                for i,c in enumerate(centers,1):
                    names.extend([f'amplitude_{i}',f'center_{i}_nm',f'sigma_{i}_nm'])
                    p0.extend([amp0,c,width0])
                    lower.extend([0.0,float(wx[0]),dx/10])
                    upper.extend([np.inf,float(wx[-1]),span])
                names.append('baseline')
                p0.append(baseline0)
                lower.append(-np.inf)
                upper.append(np.inf)
                def gaussian_bands(w,*p):
                    pred=np.full_like(np.asarray(w,dtype=float),p[-1])
                    for j in range(npeak):
                        amp,center,width=p[3*j:3*j+3]
                        pred += amp*np.exp(-0.5*((w-center)/width)**2)
                    return pred
                fit=fit_custom_molecular_model(
                    wx,wy,gaussian_bands,parameter_names=names,initial_values=p0,
                    bounds=(lower,upper))
                st.session_state['tis_molecular_gaussian_fit']={
                    'data_sha':sha,'xcol':xcol,'ycol':ycol,'centers':centers,'wavelength_nm':wx.tolist(),'measured':wy.tolist(),'result':fit}
            except Exception as e:
                st.error(str(e))
        saved_fit=st.session_state.get('tis_molecular_gaussian_fit')
        if saved_fit and all(saved_fit.get(k)==v for k,v in {'data_sha':sha,'xcol':xcol,'ycol':ycol,'centers':[float(v.strip()) for v in centers_text.split(',') if v.strip()] if centers_text.strip() else []}.items()):
            _fit=saved_fit['result']
            st.json({k:v for k,v in _fit.items() if k not in {'fitted_spectrum','residual','covariance'}})
            st.line_chart(pd.DataFrame({'measured':saved_fit['measured'],'fitted':_fit['fitted_spectrum'],'residual':_fit['residual']},index=saved_fit['wavelength_nm']))
            json_download('Export empirical molecular band fit',_fit,'tis_molecular_gaussian_fit.json')
    elif mode=='Molecular Boltzmann temperature':
        st.caption('Fit an excitation temperature from a supplied molecular transition table. Use response-corrected, background-subtracted integrated energy intensities and authoritative transition data; this is not automatic species identification or a Te/Ti measurement.')
        st.info('Required line-list columns: upper-state energy (eV), integrated line intensity, Einstein A (s⁻¹), upper-state degeneracy, and wavelength (nm). The fit assumes compatible transitions, optically thin emission and a Boltzmann population distribution.')
        energy_col=col_select('Upper-state energy (eV)',numeric,'mbt_energy')
        intensity_col=col_select('Integrated line intensity (energy units)',numeric,'mbt_intensity')
        einstein_col=col_select('Einstein A coefficient (s⁻¹)',numeric,'mbt_einstein')
        degeneracy_col=col_select('Upper-state statistical weight',numeric,'mbt_degeneracy')
        wavelength_col=col_select('Transition wavelength (nm)',numeric,'mbt_wavelength')
        uncertainty_col=st.selectbox('Integrated-intensity uncertainty σ (optional)', ['<none>']+numeric, key='mbt_uncertainty')
        if st.button('Fit molecular excitation temperature',key='mbt_fit_button'):
            try:
                columns={
                    'energy':pd.to_numeric(df[energy_col],errors='coerce'),
                    'intensity':pd.to_numeric(df[intensity_col],errors='coerce'),
                    'einstein_a':pd.to_numeric(df[einstein_col],errors='coerce'),
                    'degeneracy':pd.to_numeric(df[degeneracy_col],errors='coerce'),
                    'wavelength':pd.to_numeric(df[wavelength_col],errors='coerce'),
                }
                if uncertainty_col!='<none>':
                    columns['uncertainty']=pd.to_numeric(df[uncertainty_col],errors='coerce')
                line_data=pd.DataFrame(columns).dropna()
                if line_data.empty:
                    raise ValueError('No complete finite transition rows remain after parsing the selected columns.')
                fit_result=fit_molecular_boltzmann_temperature(
                    line_data['energy'].to_numpy(dtype=float),
                    line_data['intensity'].to_numpy(dtype=float),
                    line_data['einstein_a'].to_numpy(dtype=float),
                    line_data['degeneracy'].to_numpy(dtype=float),
                    line_data['wavelength'].to_numpy(dtype=float),
                    intensity_uncertainty=(line_data['uncertainty'].to_numpy(dtype=float) if uncertainty_col!='<none>' else None),
                )
                fit_result['input_columns']={
                    'upper_energy_eV':energy_col,'integrated_intensity':intensity_col,
                    'einstein_A_s':einstein_col,'degeneracy':degeneracy_col,
                    'wavelength_nm':wavelength_col,'intensity_uncertainty':uncertainty_col,
                }
                ordinate=np.log(line_data['intensity'].to_numpy(dtype=float)*line_data['wavelength'].to_numpy(dtype=float)/(line_data['einstein_a'].to_numpy(dtype=float)*line_data['degeneracy'].to_numpy(dtype=float)))
                st.session_state['tis_molecular_boltzmann_fit']={
                    'data_sha':sha,'columns':fit_result['input_columns'],
                    'energy':line_data['energy'].tolist(),'ordinate':ordinate.tolist(),'result':fit_result,
                }
            except Exception as e:
                st.error(str(e))
        saved_boltzmann=st.session_state.get('tis_molecular_boltzmann_fit')
        selected_columns={
            'upper_energy_eV':energy_col,'integrated_intensity':intensity_col,
            'einstein_A_s':einstein_col,'degeneracy':degeneracy_col,
            'wavelength_nm':wavelength_col,'intensity_uncertainty':uncertainty_col,
        }
        if saved_boltzmann and saved_boltzmann.get('data_sha')==sha and saved_boltzmann.get('columns')==selected_columns:
            _br=saved_boltzmann['result']
            st.json({k:v for k,v in _br.items() if k not in {'fitted_log_population_over_weight','residual'}})
            st.line_chart(pd.DataFrame({
                'log population / weight':saved_boltzmann['ordinate'],
                'Boltzmann fit':_br['fitted_log_population_over_weight'],
                'residual':_br['residual'],
            },index=saved_boltzmann['energy']))
            json_download('Export molecular Boltzmann temperature',_br,'tis_molecular_boltzmann_temperature.json')
    elif mode=='Line-ratio analysis':
        st.caption('Enter background-subtracted line intensities and relative instrument responses. Atomic modelling is needed to infer Te, ne or impurity concentration.')
        c1,c2=st.columns(2)
        with c1: i1=st.number_input('Line 1 intensity',min_value=0.0,value=1000.0); r1=st.number_input('Relative response at line 1',min_value=0.000001,value=1.0)
        with c2: i2=st.number_input('Line 2 intensity',min_value=0.000001,value=500.0); r2=st.number_input('Relative response at line 2',min_value=0.000001,value=1.0)
        if st.button('Calculate corrected line ratio'):
            try: result=corrected_line_ratio(i1,i2,r1,r2); st.json(result); json_download('Export line-ratio result',result,'tis_line_ratio.json')
            except Exception as e: st.error(str(e))
    elif mode=='Wavelength calibration':
        rawcol=col_select('Raw pixel / coordinate',numeric,'cal_raw'); pix=st.text_input('Reference pixel coordinates','100,500,900'); wav=st.text_input('Reference wavelengths','500,550,600'); deg=st.slider('Polynomial degree',1,5,1)
        if st.button('Calculate calibration'):
            try:
                r=spectral_calibration([float(x) for x in pix.split(',')],[float(x) for x in wav.split(',')],deg); st.json(r)
                calibrated=polynomial_calibration(df[rawcol],[float(v) for v in r['coefficients_high_to_low']]); out=pd.DataFrame({'raw_coordinate':df[rawcol],'calibrated_wavelength':calibrated}); st.dataframe(out.head(20)); download_df('Download calibrated spectrum coordinates',out,'tis_calibrated_wavelength.csv'); json_download('Export calibration fit JSON',r,'tis_wavelength_calibration.json')
            except Exception as e: st.error(str(e))
    else:
        energy_unit=st.radio("Energy-axis unit",["keV","eV"],horizontal=True,key="xray_energy_unit")
        ecol=col_select(f'Energy coordinate ({energy_unit})',numeric,'energy_x'); ccol=col_select('Counts / intensity',numeric,'energy_y'); bgcol=st.selectbox('Background column (optional)',['<none>']+numeric,key='energy_bg')
        representation=st.radio("Interpretation of spectral values",["Counts per bin","Signal density per energy unit"],horizontal=True,key="spectrum_representation")
        sigma_col=st.selectbox("Per-bin uncertainty σ (optional)",["<none>"]+numeric,key="spectrum_sigma")
        assume_poisson=st.checkbox("Infer Poisson uncertainty from raw counts",value=False,key="spectrum_poisson",disabled=(bgcol!="<none>" and sigma_col=="<none>"))
        bin_representation="counts_per_bin" if representation=="Counts per bin" else "density_per_eV"
        sigma_values=None if sigma_col=="<none>" else df[sigma_col]
        background_values=None if bgcol=="<none>" else df[bgcol]
        if st.button('Analyze energy spectrum'):
            try:
                if mode=='X-ray energy spectrum':
                    xray_energy=np.asarray(df[ecol],dtype=float)
                    if energy_unit=="eV": xray_energy=xray_energy/1000.0
                    result=xray_spectrum_summary(xray_energy,df[ccol],background_values,
                        count_uncertainty=sigma_values,assume_poisson_counts=assume_poisson,
                        bin_representation=bin_representation)
                    result["input_energy_axis_unit"]=energy_unit
                else:
                    generic_energy=np.asarray(df[ecol],dtype=float)
                    if energy_unit=="keV": generic_energy=generic_energy*1000.0
                    result=spectrum_moments(generic_energy,df[ccol],background_values,
                        count_uncertainty=sigma_values,assume_poisson_counts=assume_poisson,
                        bin_representation=bin_representation)
                st.json(result); json_download('Export spectrum summary',result,'tis_energy_spectrum_summary.json')
                plot_line(df[ecol],df[ccol],ecol,ccol,'Energy spectrum')
            except Exception as e: st.error(str(e))

elif page=='Probe Analysis':
    st.title('Langmuir, Ball-Pen & Magnetic Probe Analysis'); need_data()
    with st.expander('Poloidal-field coils / magnetic probes', expanded=True):
        st.caption('Calibrated signal analysis only. Supply gain/offset from the documented probe and acquisition-chain calibration; this workflow does not reconstruct poloidal flux or equilibrium.')
        time_col=col_select('Time column (s)',numeric,'pf_probe_time')
        pf_channels=st.multiselect('Poloidal-field probe signal channel(s)',[c for c in numeric if c != time_col],default=[],key='pf_probe_channels')
        st.caption('Enter one comma-separated value for all channels, or one value per selected channel, in the same order.')
        c1,c2,c3=st.columns(3)
        with c1: pf_gain_text=st.text_input('Calibration gain (T / digitizer unit)','1.0',key='pf_gain')
        with c2: pf_offset_text=st.text_input('Calibration offset (T)','0.0',key='pf_offset')
        with c3: pf_unc_text=st.text_input('Signal uncertainty 1σ (digitizer units; optional)','',key='pf_uncertainty')
        use_baseline=st.checkbox('Subtract mean baseline over a time interval',value=False,key='pf_use_baseline')
        baseline_interval=None
        if use_baseline:
            tmin=float(pd.to_numeric(df[time_col],errors='coerce').min())
            tmax=float(pd.to_numeric(df[time_col],errors='coerce').max())
            if not np.isfinite(tmin) or not np.isfinite(tmax) or tmin >= tmax:
                st.warning('The selected time column must have a finite, nonzero range.')
            else:
                baseline_interval=st.slider('Baseline interval (s)',min_value=tmin,max_value=tmax,value=(tmin,tmin+(tmax-tmin)*0.1),key='pf_baseline_interval')
        if st.button('Analyze poloidal-field probes',type='primary',key='analyze_pf_probes'):
            try:
                if not pf_channels:
                    raise ValueError('Select at least one magnetic-probe signal channel.')
                def _parse_pf_values(raw, label, count):
                    values=[float(part.strip()) for part in raw.split(',') if part.strip()]
                    if len(values)==1: values=values*count
                    if len(values)!=count or not np.all(np.isfinite(values)):
                        raise ValueError(f'{label} must be one finite value or exactly {count} comma-separated finite values.')
                    return values[0] if count==1 else values
                gain_value=_parse_pf_values(pf_gain_text,'Calibration gain',len(pf_channels))
                offset_value=_parse_pf_values(pf_offset_text,'Calibration offset',len(pf_channels))
                uncertainty_value=None
                if pf_unc_text.strip():
                    uncertainty_value=_parse_pf_values(pf_unc_text,'Signal uncertainty',len(pf_channels))
                time_values=pd.to_numeric(df[time_col],errors='coerce').to_numpy(dtype=float)
                signal_values=df[pf_channels].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
                result=analyze_poloidal_field_probes(
                    time_values,signal_values,
                    calibration_gain_T_per_unit=gain_value,
                    calibration_offset_T=offset_value,
                    signal_uncertainty=uncertainty_value,
                    baseline_interval_s=baseline_interval,
                    probe_names=pf_channels,
                )
                st.success('Poloidal-field probe analysis completed for the selected calibrated-signal scope.')
                st.json(result)
                for channel_index,channel_name in enumerate(pf_channels):
                    st.markdown(f'**{channel_name}**')
                    plot_line(result['time_s'],result['B_poloidal_T'][:,channel_index],time_col,'Bp (T)',f'{channel_name}: calibrated poloidal field')
                    plot_line(result['time_s'],result['dB_poloidal_dt_T_s'][:,channel_index],time_col,'dBp/dt (T/s)',f'{channel_name}: time derivative')
                json_download('Export poloidal-field analysis JSON',result,'tis_poloidal_field_probe_analysis.json')
            except Exception as e:
                st.error(f'Poloidal-field probe analysis failed: {e}')
        st.info('Interpretation limits: gain/offset calibration uncertainty, probe transfer function, bandwidth, geometry, timing uncertainty and correlated drift are not automatically inferred. Numerical differentiation can amplify noise; sampling diagnostics alone do not rule out aliasing.')
    st.divider()
    st.markdown('### Langmuir probe I–V analysis')
    vcol=col_select('Probe voltage (V)',numeric,'probe_v'); icol=col_select('Probe current (A)',numeric,'probe_i')
    vmin=float(pd.to_numeric(df[vcol],errors='coerce').min()); vmax=float(pd.to_numeric(df[vcol],errors='coerce').max())
    st.markdown('### Fit-window and probe geometry')
    st.caption('Fit windows are experiment-specific and are not inferred automatically. Enable each fit only after selecting a physically justified interval from the measured I–V curve. Default analysis reports descriptive quantities only.')
    efit=None
    if st.checkbox('Enable electron-retarding exponential fit (Te)',value=False,key='probe_enable_electron_fit'):
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin>=vmax:
            st.error('The selected voltage column must contain a finite, nonzero voltage range.')
        else:
            efit=st.slider('Electron-retarding fit range (V)',min_value=float(vmin),max_value=float(vmax),value=(float(vmin+(vmax-vmin)*.35),float(vmin+(vmax-vmin)*.75)),key='probe_electron_fit_range')
    ifit=None
    if st.checkbox('Enable ion-branch fit (required for density estimate)',value=False,key='probe_enable_ion_fit'):
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin>=vmax:
            st.error('The selected voltage column must contain a finite, nonzero voltage range.')
        else:
            ifit=st.slider('Ion-branch fit range (V)',min_value=float(vmin),max_value=float(vmax),value=(float(vmin),float(vmin+(vmax-vmin)*.2)),key='probe_ion_fit_range')
    c1,c2,c3=st.columns(3)
    with c1: area=st.number_input('Effective collection area (m²; 0=omit density)',min_value=0.0,value=0.0,format='%.8g')
    with c2: mass=st.number_input('Main ion mass (amu; 0 = do not estimate density)',min_value=0.0,value=0.0)
    with c3: ti=st.number_input('Ion temperature (eV; model input)',min_value=0.0,value=0.0)
    sheath=st.number_input('Sheath coefficient',min_value=0.01,value=.61)
    cur_conv=st.radio('Current sign convention',['electron_positive','ion_positive'],horizontal=True,key='probe_current_convention',
        help='electron_positive: ion-saturation current is negative (default). ion_positive: collected ion current is positive; it is flipped internally.')
    if st.button('Analyze probe I–V',type='primary'):
        try:
            result=langmuir_iv_analysis(df[vcol],df[icol],efit,ifit,area or None,mass or None,ti,sheath,current_convention=cur_conv)
            st.json(result); plot_line(df[vcol],df[icol],vcol,icol,'Measured probe I–V'); json_download('Export probe analysis JSON',result,'tis_probe_analysis.json')
        except Exception as e: st.error(str(e))
    st.warning('Default output is descriptive only. Te and density are calculated only when their fit windows are explicitly enabled; inspect fit quality and confirm calibration, sweep polarity, saturation regions, magnetic-field effects and the probe/sheath model before interpreting results.')

elif page=='Plasma Parameters':
    st.title('Plasma Parameters & Diagnostic Conversions')
    tabs=st.tabs(['Derived parameters','Interferometry','Reflectometry / ECE','Turbulent flux','Neutron rate'])
    with tabs[0]:
        a,b,c=st.columns(3)
        with a: ne=st.number_input('Electron density nₑ (m⁻³)',min_value=0.0,value=1e19,format='%.6g'); ni=st.number_input('Ion density nᵢ (m⁻³; 0=quasineutral estimate)',min_value=0.0,value=0.0,format='%.6g'); te=st.number_input('Electron temperature Tₑ (eV)',min_value=0.0,value=1000.0); ti=st.number_input('Ion temperature Tᵢ (eV)',min_value=0.0,value=800.0)
        with b: ip=st.number_input('Signed plasma current Iₚ (A)',value=50000.0); vl=st.number_input('Loop voltage (V)',value=1.0); R=st.number_input('Major radius R (m)',min_value=1e-9,value=.68); minor=st.number_input('Minor radius a (m)',min_value=1e-9,value=.12)
        with c: bt=st.number_input('Signed toroidal field Bₜ (T)',value=1.0); mi=st.number_input('Main ion mass (amu)',min_value=1e-6,value=2.0); W=st.number_input('Stored thermal energy (J; 0=omit)',min_value=0.0,value=0.0); ploss=st.number_input('Loss power (W; 0=omit)',min_value=0.0,value=0.0)
        try: result=derive_parameters(ne,te,ti,ip,vl,R,minor,bt,W or None,ploss or None,ni or None,mi); st.json(result); json_download('Export parameter calculations',result,'tis_plasma_parameters.json')
        except Exception as e: st.error(str(e))
    with tabs[1]:
        ph=st.number_input('Interferometer phase shift (rad)',value=.1); freq=st.number_input('Probe frequency (Hz)',min_value=1.0,value=140e9,format='%.8g'); phase_sign=st.selectbox('Calibrated instrument phase convention',[-1,1],index=None,placeholder='Select measured convention')
        if st.button('Convert phase to line-integrated density'):
            if phase_sign is None:
                st.error('Select the sign convention from the instrument documentation/calibration; TIS will not assume a sign.')
            else:
                density=float(linear_phase_density(ph,freq,sign_convention=phase_sign))
                if density<0: st.warning('The selected convention and phase produce a negative line-integrated density. Check phase sign, reference phase and calibration before interpretation.')
                st.metric('Approximate line-integrated density (m⁻²)',f'{density:.7g}')
        st.caption('Output is ∫nₑ dl in m⁻², not local density. Sign convention must be selected from instrument calibration; negative results are flagged.')
    with tabs[2]:
        st.markdown('#### O-mode reflectometry cutoff density')
        rf=st.number_input('Probe frequency (GHz)',min_value=0.001,value=60.0,key='refl_freq')
        if st.button('Calculate O-mode cutoff density'):
            try: st.json(reflectometry_cutoff_density(rf*1e9,'O'))
            except Exception as e: st.error(str(e))
        st.markdown('#### Electron-cyclotron resonance field')
        ef=st.number_input('ECE frequency (GHz)',min_value=0.001,value=110.0,key='ece_freq'); harm=st.selectbox('Harmonic',[1,2,3],index=1)
        if st.button('Calculate resonance field'):
            try: st.json(electron_cyclotron_resonance_field(ef*1e9,harm))
            except Exception as e: st.error(str(e))
        st.divider()
        st.markdown('#### ECE channel analysis')
        st.caption('Use calibrated spectral radiance for brightness temperature. Temperature is not automatically electron temperature: optical thickness and thermalization must be justified independently.')
        ece_mode=st.selectbox('ECE workflow',['Brightness temperature','Channel fluctuations / PSD','Resonance mapping'],key='ece_workflow_mode')
        ece_freq_text=st.text_input('Channel center frequencies (GHz, comma-separated, ascending)',value='80, 90, 100',key='ece_channel_freqs')
        try:
            ece_freq_GHz=np.asarray([float(v.strip()) for v in ece_freq_text.split(',') if v.strip()],dtype=float)
            if ece_freq_GHz.ndim!=1 or ece_freq_GHz.size==0 or np.any(~np.isfinite(ece_freq_GHz)) or np.any(ece_freq_GHz<=0) or np.any(np.diff(ece_freq_GHz)<=0):
                raise ValueError('Enter finite, positive, strictly increasing channel frequencies.')
            ece_freq_Hz=ece_freq_GHz*1e9
            if ece_mode=='Brightness temperature':
                ece_rad_text=st.text_input('Calibrated spectral radiance per channel (W m⁻² sr⁻¹ Hz⁻¹; comma-separated)',key='ece_radiance')
                ece_bandwidth_GHz=st.number_input('Channel bandwidth (GHz; 0 = not supplied)',min_value=0.0,value=0.0,step=0.1,key='ece_bandwidth_ghz')
                if st.button('Analyze ECE brightness temperature',type='primary',key='ece_run_brightness'):
                    ece_rad=np.asarray([float(v.strip()) for v in ece_rad_text.split(',') if v.strip()],dtype=float)
                    if ece_rad.shape!=ece_freq_Hz.shape:
                        raise ValueError('Radiance count must equal the number of channel frequencies.')
                    ece_kwargs={}
                    if ece_bandwidth_GHz>0:
                        ece_kwargs['channel_bandwidth_Hz']=ece_bandwidth_GHz*1e9
                    result=analyze_ece(ece_freq_Hz,harmonic=int(harm),spectral_radiance_W_m2_sr_Hz=ece_rad,mode='brightness_temperature',**ece_kwargs)
                    st.success('ECE brightness-temperature analysis completed.')
                    st.dataframe(pd.DataFrame({'Frequency (GHz)':ece_freq_GHz,'Resonance field (T)':result['resonance_field_T'],'Brightness temperature (K)':result['brightness_temperature_K'],'Brightness temperature (eV)':result['brightness_temperature_eV']}),use_container_width=True,hide_index=True)
                    st.line_chart(pd.DataFrame({'Brightness temperature (K)':np.asarray(result['brightness_temperature_K']).reshape(-1),'Brightness temperature (eV)':np.asarray(result['brightness_temperature_eV']).reshape(-1)},index=ece_freq_GHz))
                    st.json(result)
                    json_download('Export ECE brightness-temperature result',result,'tis_ece_brightness_temperature.json')
            elif ece_mode=='Channel fluctuations / PSD':
                if df is None:
                    st.info('Upload a table containing a time column and one calibrated signal column per ECE channel.')
                else:
                    ece_time_col=col_select('ECE time column (s)',list(df.columns),'ece_time_col')
                    ece_signal_cols=st.multiselect('ECE channel signal columns (same order as frequencies)',[c for c in numeric if c!=ece_time_col],key='ece_signal_cols')
                    ece_noise=st.number_input('Per-channel RMS noise estimate (signal units; 0 = omit)',min_value=0.0,value=0.0,key='ece_noise_std')
                    if st.button('Analyze ECE fluctuations / PSD',type='primary',key='ece_run_fluctuations'):
                        if len(ece_signal_cols)!=len(ece_freq_Hz):
                            raise ValueError('Select exactly one signal column per channel frequency.')
                        ece_frame=df[[ece_time_col]+ece_signal_cols].apply(pd.to_numeric,errors='coerce').dropna()
                        ece_t=ece_frame[ece_time_col].to_numpy(dtype=float)
                        ece_y=ece_frame[ece_signal_cols].to_numpy(dtype=float)
                        ece_kwargs={'noise_std':ece_noise} if ece_noise>0 else {}
                        result=analyze_ece(ece_freq_Hz,harmonic=int(harm),signal=ece_y,time_s=ece_t,mode='channel_fluctuations',**ece_kwargs)
                        st.success('ECE fluctuation analysis completed.')
                        st.metric('Sampling frequency',f"{result['sampling_frequency_Hz']:.6g} Hz")
                        st.metric('Nyquist frequency',f"{result['nyquist_frequency_Hz']:.6g} Hz")
                        rms_frame=pd.DataFrame({'Frequency (GHz)':ece_freq_GHz,'Fluctuation RMS':result['fluctuation_rms']})
                        st.markdown('**Per-channel fluctuation RMS**')
                        st.dataframe(rms_frame,use_container_width=True,hide_index=True)
                        st.bar_chart(rms_frame.set_index('Frequency (GHz)')['Fluctuation RMS'])
                        psd_frame=pd.DataFrame(result['fluctuation_psd'],index=result['fluctuation_psd_frequency_Hz'],columns=[f'{v:g} GHz' for v in ece_freq_GHz])
                        psd_frame.index.name='Frequency (Hz)'
                        st.markdown('**One-sided fluctuation power spectral density**')
                        st.line_chart(psd_frame)
                        if 'above_noise_rms' in result:
                            st.dataframe(pd.DataFrame({'Frequency (GHz)':ece_freq_GHz,'RMS above noise':result['above_noise_rms']}),use_container_width=True,hide_index=True)
                        st.json({k:v for k,v in result.items() if k not in {'fluctuation_psd','fluctuation_psd_frequency_Hz','time_s'}})
                        json_download('Export ECE fluctuation / PSD result',result,'tis_ece_fluctuations.json')
            else:
                ece_r0=st.number_input('Reference major radius R₀ (m)',min_value=1e-6,value=1.0,key='ece_r0')
                ece_b0=st.number_input('Toroidal field at R₀ (T)',min_value=1e-6,value=2.0,key='ece_b0')
                if st.button('Map ECE resonance radii',type='primary',key='ece_run_mapping'):
                    result=analyze_ece(ece_freq_Hz,harmonic=int(harm),major_radius_m=ece_r0,B_axis_T=ece_b0,mode='resonance_mapping')
                    st.success('ECE resonance mapping completed using the declared circular toroidal-field approximation.')
                    st.dataframe(pd.DataFrame({'Frequency (GHz)':ece_freq_GHz,'Resonance field (T)':result['resonance_field_T'],'Approximate major radius (m)':result['resonance_radius_m']}),use_container_width=True,hide_index=True)
                    st.line_chart(pd.DataFrame({'Approximate major radius (m)':result['resonance_radius_m']},index=ece_freq_GHz))
                    st.warning('This is not EFIT mapping; plasma shaping, poloidal field, relativistic/Doppler shifts and line-of-sight geometry are not included.')
                    st.json(result)
                    json_download('Export ECE resonance mapping',result,'tis_ece_resonance_mapping.json')
        except Exception as e:
            st.error(str(e))
    with tabs[3]:
        st.write('For aligned columns, calculate covariance fluxes from calibrated density, radial velocity and optional temperature.')
        if df is not None and len(numeric)>=2:
            ncol=col_select('Density fluctuation column (m⁻³)',numeric,'flux_n'); vrcol=col_select('Radial velocity column (m/s)',[x for x in numeric if x!=ncol],'flux_v'); tcol2=st.selectbox('Temperature column in eV (optional)',['<none>']+[x for x in numeric if x not in [ncol,vrcol]],key='flux_t')
            if st.button('Calculate fluctuation flux'):
                try: result=fluctuation_flux(df[ncol],df[vrcol],None if tcol2=='<none>' else df[tcol2]); st.json(result); json_download('Export fluctuation flux',result,'tis_fluctuation_flux.json')
                except Exception as e: st.error(str(e))
        else: st.info('Load aligned density and velocity columns to calculate fluxes.')
    with tabs[4]:
        counts=st.number_input('Measured count rate (s⁻¹)',min_value=0.0,value=1000.0); eff=st.number_input('Detection efficiency (0–1]',min_value=.000001,max_value=1.0,value=.1); geom=st.number_input('Geometric / solid-angle factor',min_value=.000001,value=1.0); cal=st.number_input('Additional calibration factor',min_value=.000001,value=1.0)
        if st.button('Estimate neutron source rate'):
            try: result=neutron_rate_from_counts(counts,eff,geom,cal); st.json(result); json_download('Export neutron estimate',result,'tis_neutron_rate.json')
            except Exception as e: st.error(str(e))

elif page=='Equilibrium & Reconstruction':
    st.title('Equilibrium & Magnetic Reconstruction Tools')
    st.warning('This page analyzes a supplied equilibrium flux grid. It does not solve the Grad–Shafranov equation or replace EFIT / a validated equilibrium reconstruction code.')
    if df is not None: st.info('For grid data, upload separate arrays as CSV files below. R and Z must be physical coordinates in metres.')
    Rfile=st.file_uploader('R coordinate array CSV',type=['csv'],key='eq_R'); Zfile=st.file_uploader('Z coordinate array CSV',type=['csv'],key='eq_Z'); Pfile=st.file_uploader('Poloidal-flux grid ψ(R,Z) CSV',type=['csv'],key='eq_P')
    bt=st.number_input('Toroidal field input Bφ (T)',value=1.0)
    if df is not None and 'z_m' in df.columns:
        st.markdown('**Or use the loaded table as the grid** (format: `z_m` plus one `R=…_m` column per radius, as exported by TIS).')
        is_psi=st.checkbox('I confirm the loaded grid is poloidal flux ψ(R,Z) (NOT a current-density or other field)',value=False,key='eq_grid_is_psi')
        if st.button('Calculate field from loaded grid',disabled=not is_psi,key='eq_from_table'):
            try:
                g=grid_from_table(df); out=equilibrium_grid_derivatives(g['R_m'],g['Z_m'],g['grid'],bt)
                st.json({'grid_shape':list(g['grid'].shape),'Bpol_min_T':float(np.nanmin(out['Bp_T'])),'Bpol_max_T':float(np.nanmax(out['Bp_T'])),'note':out['note']})
            except Exception as e: st.error(str(e))
    if Rfile and Zfile and Pfile and st.button('Calculate magnetic-field grid'):
        try:
            Rv=pd.read_csv(Rfile,header=None).to_numpy(float).ravel(); Zv=pd.read_csv(Zfile,header=None).to_numpy(float).ravel(); psi=pd.read_csv(Pfile,header=None).to_numpy(float)
            out=equilibrium_grid_derivatives(Rv,Zv,psi,bt); st.json({'grid_shape':list(psi.shape),'Bpol_min_T':float(np.nanmin(out['Bp_T'])),'Bpol_max_T':float(np.nanmax(out['Bp_T'])),'Btotal_max_T':float(np.nanmax(out['Btotal_T'])),'note':out['note']})
            rows=[]
            for key in ['BR_T','BZ_T','Bp_T','Btotal_T']:
                rows.append(pd.DataFrame(out[key],index=Zv,columns=Rv).stack().rename(key))
            flat=pd.concat(rows,axis=1).reset_index(names=['Z_m','R_m']); st.dataframe(flat.head(20)); download_df('Download derived magnetic field grid',flat,'tis_equilibrium_field_grid.csv')
        except Exception as e: st.error(str(e))
    st.markdown('### Geometry / profile integration')
    if df is not None and len(numeric)>=2:
        xc=col_select('Profile coordinate',numeric,'eq_x'); yc=col_select('Profile value',numeric,'eq_y')
        if st.button('Integrate profile'): 
            try: res=integrate_profile(df[xc],df[yc]); st.json(res); json_download('Export profile integral',res,'tis_profile_integral.json')
            except Exception as e: st.error(str(e))

elif page=='Tomography & Bolometry':
    st.title('Tomography & Bolometry')
    tomo_mode=st.radio('Reconstruction workflow',['General response-matrix inversion','Build 2-D ray geometry matrix','Axisymmetric Abel inversion','Axisymmetric radiated-power integration'],horizontal=True)
    lam=st.number_input('Tikhonov regularization λ',min_value=0.0,value=.01,format='%.8g',key='tomo_lam'); nonneg=st.checkbox('Constrain reconstructed profile to be nonnegative',value=True,key='tomo_nonneg')
    if tomo_mode=='General response-matrix inversion':
        st.write('Solve y = A x + noise using a calibrated geometry / response matrix.')
        Afile=st.file_uploader('Response / geometry matrix A (CSV, no header)',type=['csv'],key='tomo_A'); yfile=st.file_uploader('Measurements y (CSV, one column)',type=['csv'],key='tomo_y'); sigfile=st.file_uploader('Measurement standard uncertainties σ (optional CSV column)',type=['csv'],key='tomo_sigma')
        auto_lambda=st.checkbox('Select λ using an L-curve curvature heuristic',value=False,key='tomo_auto_lambda')
        use_spatial_2d=st.checkbox('Use spatial 2-D regularization for a row-major (z, x) pixel grid',value=False,key='tomo_spatial_2d')
        grid_shape=None
        if use_spatial_2d:
            nz=st.number_input('Grid rows (nz)',min_value=1,value=10,step=1,key='tomo_nz')
            nx=st.number_input('Grid columns (nx)',min_value=1,value=10,step=1,key='tomo_nx')
            grid_shape=(int(nz),int(nx))
            st.caption('The grid must contain exactly nz × nx unknowns, flattened as iz * nx + ix. Spatial smoothing uses horizontal and vertical neighbors.')
        if Afile and yfile and st.button('Run regularized reconstruction',type='primary'):
            try:
                A=pd.read_csv(Afile,header=None).to_numpy(float); y=pd.read_csv(yfile,header=None).to_numpy(float).ravel()
                sigma=None if sigfile is None else pd.read_csv(sigfile,header=None).to_numpy(float).ravel()
                lambda_used=lam
                if auto_lambda:
                    choice=select_tikhonov_lambda(A,y,nonnegative=nonneg,measurement_sigma=sigma,grid_shape=grid_shape)
                    lambda_used=choice['selected_lambda']
                    st.metric('L-curve selected λ',f'{lambda_used:.6g}')
                    plot_line(np.log10(np.maximum(choice['weighted_residual_norms'],np.finfo(float).tiny)),np.log10(np.maximum(choice['regularization_norms'],np.finfo(float).tiny)),'log₁₀ weighted residual norm','log₁₀ regularization-operator norm','Tikhonov L-curve')
                    json_download('Export L-curve selection',choice,'tis_tomography_lcurve.json')
                r=regularized_tomography(A,y,lambda_used,nonneg,measurement_sigma=sigma,grid_shape=grid_shape)
                st.json({k:v for k,v in r.items() if k!='solution'})
                if grid_shape is not None:
                    nz, nx = grid_shape
                    out = pd.DataFrame({'iz':np.repeat(np.arange(nz), nx),
                                        'ix':np.tile(np.arange(nx), nz),
                                        'reconstructed_value':r['solution']})
                else:
                    out = pd.DataFrame({'basis_or_voxel_index':np.arange(len(r['solution'])),
                                        'reconstructed_value':r['solution']})
                st.dataframe(out)
                download_df('Download reconstruction CSV',out,'tis_tomography_solution.csv')
            except Exception as e: st.error(str(e))
    elif tomo_mode=='Build 2-D ray geometry matrix':
        st.write('Build a path-length response matrix from straight diagnostic lines of sight crossing a rectangular (x,z) pixel grid. This is geometry-based; it does not infer detector response, attenuation or equilibrium mapping.')
        ray_start=st.file_uploader('Ray start coordinates (CSV: x,z metres; no header)',type=['csv'],key='tomo_ray_start')
        ray_end=st.file_uploader('Ray end coordinates (CSV: x,z metres; no header)',type=['csv'],key='tomo_ray_end')
        x_edges_text=st.text_input('x pixel edges (m, comma-separated)','-1,0,1')
        z_edges_text=st.text_input('z pixel edges (m, comma-separated)','-1,0,1')
        if ray_start is not None and ray_end is not None and st.button('Build geometry response matrix',type='primary'):
            try:
                from tis.core import build_tomography_matrix_2d
                starts=pd.read_csv(ray_start,header=None).to_numpy(float)
                ends=pd.read_csv(ray_end,header=None).to_numpy(float)
                x_edges=[float(v.strip()) for v in x_edges_text.split(',')]
                z_edges=[float(v.strip()) for v in z_edges_text.split(',')]
                matrix=build_tomography_matrix_2d(starts,ends,x_edges,z_edges)
                out=pd.DataFrame(matrix,columns=[f'pixel_{i}' for i in range(matrix.shape[1])])
                st.success(f'Built {matrix.shape[0]} rays × {matrix.shape[1]} pixels; entries are path lengths in metres.')
                st.dataframe(out,use_container_width=True)
                download_df('Download geometry response matrix A',out,'tis_tomography_geometry_matrix.csv')
                json_download('Export geometry matrix metadata',{'matrix_shape':list(matrix.shape),'x_edges_m':x_edges,'z_edges_m':z_edges,'column_order':'iz * nx + ix','entry_units':'m path length','model_limit':'Straight rays through rectangular pixels only; detector response, attenuation, toroidal geometry and equilibrium mapping are not included.'},'tis_tomography_geometry_manifest.json')
            except Exception as e: st.error(str(e))
    elif tomo_mode=='Axisymmetric Abel inversion':
        btxt=st.text_input('Chord impact parameters b (m, comma-separated)','0,0.02,0.04,0.06,0.08'); ytxt=st.text_input('Line-integrated measurements (units·m, comma-separated)','1.0,0.9,0.7,0.4,0.1'); etxt=st.text_input('Radial bin edges (m, comma-separated)','0,0.02,0.04,0.06,0.08,0.10')
        sigmatxt=st.text_input('Measurement standard uncertainties σ (optional, same length; comma-separated)','')
        if st.button('Run Abel inversion',type='primary'):
            try:
                bvals=[float(z) for z in btxt.split(',')]
                yvals=[float(z) for z in ytxt.split(',')]
                edges=[float(z) for z in etxt.split(',')]
                sigma_vals=None if not sigmatxt.strip() else [float(z) for z in sigmatxt.split(',')]
                r=abel_reconstruction(bvals,yvals,edges,lam,nonneg,measurement_sigma=sigma_vals); out=pd.DataFrame({'radius_m':r['radius_centers_m'],'reconstructed_profile':r['emissivity_or_density_profile']}); st.json({k:v for k,v in r.items() if k not in ['response_matrix','radius_centers_m','emissivity_or_density_profile']}); st.dataframe(out); plot_line(out['radius_m'],out['reconstructed_profile'],'Radius (m)','Reconstructed value','Abel inversion'); download_df('Download Abel profile',out,'tis_abel_profile.csv')
            except Exception as e: st.error(str(e))
    else:
        Rf=st.file_uploader('Major-radius grid R (m), CSV row',type=['csv'],key='rad_R'); Zf=st.file_uploader('Vertical grid Z (m), CSV column',type=['csv'],key='rad_Z'); Ef=st.file_uploader('Emissivity grid (W/m³), CSV',type=['csv'],key='rad_E')
        if Rf and Zf and Ef and st.button('Integrate total radiated power'):
            try:
                r=axisymmetric_radiated_power(pd.read_csv(Rf,header=None).to_numpy(float).ravel(),pd.read_csv(Zf,header=None).to_numpy(float).ravel(),pd.read_csv(Ef,header=None).to_numpy(float)); st.json(r); json_download('Export radiated power estimate',r,'tis_radiated_power.json')
            except Exception as e: st.error(str(e))
    st.markdown('### Target heat-flux profile fit')
    if df is not None and len(numeric)>=2:
        xcol=col_select('Target coordinate (m)',numeric,'heat_x'); qcol=col_select('Heat flux (W/m²)',numeric,'heat_q')
        heat_model=st.selectbox('Heat-flux model',['Eich-like convolved profile','Simple exponential'],key='heat_model')
        st.caption('The Eich-like model includes Gaussian spreading and reports fit-parameter uncertainties. Target-coordinate mapping and diagnostic response remain user-supplied.')
        if st.button('Fit heat-flux profile'):
            try:
                r=eich_heat_flux_fit(df[xcol],df[qcol]) if heat_model=='Eich-like convolved profile' else exponential_heat_flux_fit(df[xcol],df[qcol])
                st.json(r); json_download('Export heat-flux fit',r,'tis_heat_flux_fit.json')
            except Exception as e: st.error(str(e))

elif page=='Advanced Analysis':
    st.title('Advanced Diagnostic Analysis')
    workflow=st.selectbox('Select workflow',['Calibration & uncertainty','Profile integration & gradients','Fast-ion / neutron energy spectrum','Data-driven correlation matrix','Triple-probe temperature','Thomson calibrated profile analysis','Thomson raw-spectrum temperature fit','Thomson calibrated scattered-power density','Registered analysis modules'])
    if workflow=='Thomson calibrated profile analysis':
        st.markdown('Analyze already-calibrated radial electron-density and electron-temperature profiles.')
        st.caption('Inputs are calibrated profiles, not raw detector counts. Radius is in metres, density in m⁻³, and electron temperature in eV.')
        if df is None or len(numeric) < 3:
            st.info('Load a table with radius, electron-density, and electron-temperature columns.')
        else:
            radius_col=col_select('Radius / minor-radius coordinate (m)',numeric,'tis_thomson_profile_radius')
            density_col=col_select('Electron density (m⁻³)',numeric,'tis_thomson_profile_density')
            temperature_col=col_select('Electron temperature (eV)',numeric,'tis_thomson_profile_temperature')
            uncertainty_options=['(none)']+list(numeric)
            density_uncertainty_col=st.selectbox('Optional density 1-sigma uncertainty column (m⁻³)',uncertainty_options,key='tis_thomson_profile_density_unc')
            temperature_uncertainty_col=st.selectbox('Optional temperature 1-sigma uncertainty column (eV)',uncertainty_options,key='tis_thomson_profile_temperature_unc')
            use_volume_average=st.checkbox('Calculate approximate volume-weighted averages',value=False,key='tis_thomson_profile_volume')
            major_radius=None
            if use_volume_average:
                major_radius=st.number_input('Major radius (m)',min_value=0.000001,value=1.0,format='%.6f',key='tis_thomson_profile_major_radius')
            if st.button('Analyze calibrated Thomson profiles',type='primary',key='run_thomson_profile'):
                try:
                    required_cols=[radius_col,density_col,temperature_col]
                    if len(set(required_cols))!=3:
                        raise ValueError('Radius, density, and temperature must use different columns.')
                    for optional_col in (density_uncertainty_col,temperature_uncertainty_col):
                        if optional_col!='(none)' and optional_col in required_cols:
                            raise ValueError('Uncertainty columns must differ from radius, density, and temperature columns.')
                    radius_values=pd.to_numeric(df[radius_col],errors='raise').to_numpy(dtype=float)
                    density_values=pd.to_numeric(df[density_col],errors='raise').to_numpy(dtype=float)
                    temperature_values=pd.to_numeric(df[temperature_col],errors='raise').to_numpy(dtype=float)
                    density_uncertainty=None if density_uncertainty_col=='(none)' else pd.to_numeric(df[density_uncertainty_col],errors='raise').to_numpy(dtype=float)
                    temperature_uncertainty=None if temperature_uncertainty_col=='(none)' else pd.to_numeric(df[temperature_uncertainty_col],errors='raise').to_numpy(dtype=float)
                    profile=analyze_thomson_profile(
                        radius_values,density_values,temperature_values,
                        density_uncertainty_m3=density_uncertainty,
                        temperature_uncertainty_ev=temperature_uncertainty,
                        major_radius_m=major_radius,
                    )
                    summary=profile['summary']
                    st.success('Calibrated Thomson profile analysis completed.')
                    metric_cols=st.columns(2)
                    metric_cols[0].metric('Peak electron density',f"{summary.density_peak_m3:.4g} m⁻³")
                    metric_cols[1].metric('Peak electron temperature',f"{summary.temperature_peak_ev:.4g} eV")
                    profile_frame=pd.DataFrame({
                        'Radius (m)':profile['radius_m'],
                        'Electron density (m⁻³)':profile['density_m3'],
                        'Electron temperature (eV)':profile['temperature_eV'],
                        'Density gradient (m⁻⁴)':profile['density_gradient_m-4'],
                        'Temperature gradient (eV/m)':profile['temperature_gradient_eV_m-1'],
                        'Density scale length (m)':profile['density_scale_length_m'],
                        'Temperature scale length (m)':profile['temperature_scale_length_m'],
                    })
                    optional_columns={
                        'Density uncertainty (m⁻³)':'density_uncertainty_m3',
                        'Temperature uncertainty (eV)':'temperature_uncertainty_eV',
                        'Density-gradient uncertainty (m⁻⁴)':'density_gradient_uncertainty_m-4',
                        'Temperature-gradient uncertainty (eV/m)':'temperature_gradient_uncertainty_eV_m-1',
                        'Density scale-length uncertainty (m)':'density_scale_length_uncertainty_m',
                        'Temperature scale-length uncertainty (m)':'temperature_scale_length_uncertainty_m',
                    }
                    for column_label,result_key in optional_columns.items():
                        if result_key in profile:
                            profile_frame[column_label]=profile[result_key]
                    st.markdown('**Profile summary**')
                    st.json(dict(vars(summary)))
                    st.markdown('**Radial profiles**')
                    st.line_chart(profile_frame.set_index('Radius (m)')[['Electron density (m⁻³)']])
                    st.line_chart(profile_frame.set_index('Radius (m)')[['Electron temperature (eV)']])
                    st.markdown('**Gradients and local scale lengths**')
                    st.dataframe(profile_frame,use_container_width=True,hide_index=True)
                    download_df('Download Thomson profile results',profile_frame,'tis_thomson_profile_results.csv')
                    profile_json={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in profile.items() if k!='summary'}
                    profile_json['summary']=dict(vars(summary))
                    json_download('Export Thomson profile summary',profile_json,'tis_thomson_profile_summary.json')
                    st.caption('Independent pointwise uncertainties are propagated through gradients, normalized gradients, scale lengths, and averages. Peak-location uncertainty is not estimated; model and calibration uncertainties are separate.')
                except Exception as e:
                    st.error(str(e))
    elif workflow=='Thomson raw-spectrum temperature fit':
        st.markdown('Fit a calibrated Thomson-scattering spectrum with a non-relativistic Gaussian Doppler model and a linear local background.')
        st.warning('Requires a calibrated wavelength axis and known Gaussian instrument sigma. Density can be estimated only if the signal is calibrated as spectral power density in W/nm and the optical geometry is supplied.')
        if df is None or len(numeric) < 2:
            st.info('Load a table with wavelength and measured spectral-signal columns to run this analysis.')
        else:
            wavelength_col=col_select('Wavelength (nm)',numeric,'tis_thomson_wavelength_col')
            signal_col=col_select('Measured spectral signal',numeric,'tis_thomson_signal_col')
            laser_wavelength_nm=st.number_input('Laser wavelength (nm)',min_value=0.000001,value=532.0,format='%.6f',key='tis_thomson_laser_nm')
            scattering_angle_deg=st.number_input('Scattering angle (degrees)',min_value=0.001,max_value=180.0,value=90.0,format='%.3f',key='tis_thomson_angle_deg')
            instrument_sigma_nm=st.number_input('Gaussian instrument sigma (nm)',min_value=0.000001,value=0.05,format='%.6f',key='tis_thomson_instrument_sigma')
            uncertainty_options=['(none)']+list(numeric)
            uncertainty_col=st.selectbox('Optional signal standard-uncertainty column (same units as signal)',uncertainty_options,key='tis_thomson_uncertainty_col')
            spectral_power_calibrated=st.checkbox('Signal is calibrated spectral power density (W/nm); also estimate electron density',value=False,key='tis_thomson_spectral_power_calibrated')
            power_geometry={}
            propagate_geometry=False
            if spectral_power_calibrated:
                st.caption('The fitted Gaussian area is treated as collected power in W. Use this option only when the signal axis is absolutely calibrated in W/nm over the relevant spectral band.')
                incident_power=st.number_input('Incident laser power (W)',min_value=0.000001,value=2.0,format='%.6e',key='tis_thomson_spectral_incident_power')
                interaction_length=st.number_input('Effective interaction length (m)',min_value=0.000001,value=0.15,format='%.6f',key='tis_thomson_spectral_interaction_length')
                collection_solid_angle=st.number_input('Collection solid angle (sr)',min_value=0.000000001,value=1.0e-3,format='%.6e',key='tis_thomson_spectral_solid_angle')
                optical_efficiency=st.number_input('Total optical efficiency (0–1)',min_value=0.000001,max_value=1.0,value=0.7,format='%.6f',key='tis_thomson_spectral_efficiency')
                propagate_geometry=st.checkbox('Also propagate optical-geometry uncertainties',value=False,key='tis_thomson_spectral_propagate_geometry')
                power_geometry={'incident_power_W':incident_power,'interaction_length_m':interaction_length,'collection_solid_angle_sr':collection_solid_angle,'optical_efficiency':optical_efficiency}
                if propagate_geometry:
                    power_geometry.update({
                        'incident_power_uncertainty_W':st.number_input('Incident-power uncertainty (W)',min_value=0.0,value=0.02,format='%.6e',key='tis_thomson_spectral_u_incident'),
                        'interaction_length_uncertainty_m':st.number_input('Interaction-length uncertainty (m)',min_value=0.0,value=0.001,format='%.6f',key='tis_thomson_spectral_u_length'),
                        'scattering_angle_uncertainty_rad':float(np.deg2rad(st.number_input('Scattering-angle uncertainty (degrees)',min_value=0.0,value=0.5,format='%.3f',key='tis_thomson_spectral_u_angle'))),
                        'collection_solid_angle_uncertainty_sr':st.number_input('Collection-solid-angle uncertainty (sr)',min_value=0.0,value=2.0e-5,format='%.6e',key='tis_thomson_spectral_u_solid'),
                        'optical_efficiency_uncertainty':st.number_input('Optical-efficiency uncertainty',min_value=0.0,value=0.01,format='%.6f',key='tis_thomson_spectral_u_efficiency'),
                    })
            if st.button('Fit Thomson spectrum and estimate electron temperature',type='primary',key='run_thomson_spectrum_fit'):
                try:
                    if wavelength_col==signal_col:
                        raise ValueError('Wavelength and signal must use different columns.')
                    if uncertainty_col in {wavelength_col,signal_col}:
                        raise ValueError('The uncertainty column must differ from wavelength and signal columns.')
                    wavelength_values=pd.to_numeric(df[wavelength_col],errors='raise').to_numpy(dtype=float)
                    signal_values=pd.to_numeric(df[signal_col],errors='raise').to_numpy(dtype=float)
                    signal_uncertainty=None
                    if uncertainty_col!='(none)':
                        signal_uncertainty=pd.to_numeric(df[uncertainty_col],errors='raise').to_numpy(dtype=float)
                    fit=fit_thomson_spectrum(
                        wavelength_values,signal_values,
                        laser_wavelength_nm=float(laser_wavelength_nm),
                        scattering_angle_rad=float(np.deg2rad(scattering_angle_deg)),
                        instrument_sigma_nm=float(instrument_sigma_nm),
                        signal_uncertainty=signal_uncertainty,
                        spectral_power_density_calibrated=spectral_power_calibrated,
                    )
                    st.success('Thomson spectral fit completed.')
                    st.metric('Electron temperature',f"{fit['electron_temperature_eV']:.4g} eV",
                              delta=f"± {fit['electron_temperature_uncertainty_eV']:.3g} eV (fit uncertainty)")
                    st.metric('Reduced chi-square',
                              'Not available without signal uncertainties' if fit['reduced_chi_square'] is None else f"{fit['reduced_chi_square']:.4g}")
                    density_summary=None
                    if spectral_power_calibrated:
                        st.metric('Integrated collected scattered power',f"{fit['integrated_scattered_power_W']:.4g} W",
                                  delta=f"± {fit['integrated_scattered_power_uncertainty_W']:.3g} W (fit uncertainty)")
                        density_args={}
                        power_unc=float(fit['integrated_scattered_power_uncertainty_W'])
                        if np.isfinite(power_unc) and power_unc>=0:
                            density_args['scattered_power_uncertainty_W']=power_unc
                        elif propagate_geometry:
                            st.warning('The spectral fit did not provide a finite integrated-power uncertainty; only the supplied geometry uncertainties will be propagated.')
                        if not propagate_geometry:
                            # At minimum, retain the fitted-area uncertainty when it is finite.
                            pass
                        density_result=density_from_thomson_scattered_power(
                            scattered_power_W=fit['integrated_scattered_power_W'],
                            incident_power_W=power_geometry['incident_power_W'],
                            interaction_length_m=power_geometry['interaction_length_m'],
                            scattering_angle_rad=float(np.deg2rad(scattering_angle_deg)),
                            collection_solid_angle_sr=power_geometry['collection_solid_angle_sr'],
                            optical_efficiency=power_geometry['optical_efficiency'],
                            **{k:v for k,v in power_geometry.items() if k.endswith('_uncertainty_W') or k.endswith('_uncertainty_m') or k.endswith('_uncertainty_sr') or k.endswith('_uncertainty_rad') or k=='optical_efficiency_uncertainty'},
                            **density_args,
                        )
                        density_value=float(np.asarray(density_result['electron_density_m3']).item())
                        st.metric('Electron density',f'{density_value:.4g} m⁻³')
                        if 'electron_density_uncertainty_m3' in density_result:
                            st.metric('Electron-density uncertainty',f"± {float(np.asarray(density_result['electron_density_uncertainty_m3']).item()):.4g} m⁻³")
                        density_summary={k:(v.item() if isinstance(v,np.ndarray) and v.ndim==0 else v) for k,v in density_result.items()}
                    fit_frame=pd.DataFrame({
                        'Wavelength (nm)':wavelength_values,
                        'Measured signal':signal_values,
                        'Fitted spectrum':fit['fitted_spectrum'],
                        'Residual':fit['residual'],
                    })
                    st.markdown('**Measured spectrum and fitted model**')
                    st.line_chart(fit_frame.set_index('Wavelength (nm)')[['Measured signal','Fitted spectrum']])
                    st.markdown('**Fit residuals**')
                    st.line_chart(fit_frame.set_index('Wavelength (nm)')[['Residual']])
                    download_df('Download Thomson spectrum fit and residuals',fit_frame,'tis_thomson_spectrum_fit.csv')
                    summary={k:v for k,v in fit.items() if k not in {'fitted_spectrum','residual','fit_covariance'}}
                    if density_summary is not None:
                        summary['calibrated_power_density_estimate']=density_summary
                    json_download('Export Thomson fit summary',summary,'tis_thomson_spectrum_fit.json')
                    with st.expander('Model assumptions and limitations'):
                        for item in fit['assumptions']: st.markdown(f'- {item}')
                        for item in fit['limitations']: st.warning(item)
                except Exception as e:
                    st.error(str(e))
    elif workflow=='Thomson calibrated scattered-power density':
        st.markdown('Estimate electron density from calibrated collected Thomson-scattered power and a specified single-band optical geometry.')
        st.warning('Use only when collected and incident powers refer to the same calibrated spectral band and time gate. This is a single-band relation, not a full spectral-response inversion.')
        scattered_power=st.number_input('Collected scattered power (W)',min_value=0.0,value=1.0e-14,format='%.6e',key='tis_thomson_power_scattered')
        incident_power=st.number_input('Incident laser power (W)',min_value=0.000001,value=2.0,format='%.6e',key='tis_thomson_power_incident')
        interaction_length=st.number_input('Effective interaction length (m)',min_value=0.000001,value=0.15,format='%.6f',key='tis_thomson_interaction_length')
        angle_deg=st.number_input('Scattering angle (degrees)',min_value=0.001,max_value=180.0,value=90.0,format='%.3f',key='tis_thomson_power_angle')
        solid_angle=st.number_input('Collection solid angle (sr)',min_value=0.000000001,value=1.0e-3,format='%.6e',key='tis_thomson_solid_angle')
        efficiency=st.number_input('Total optical efficiency (0–1)',min_value=0.000001,max_value=1.0,value=0.7,format='%.6f',key='tis_thomson_efficiency')
        propagate=st.checkbox('Propagate independent input uncertainties',value=False,key='tis_thomson_power_propagate')
        power_uncertainty_args={}
        if propagate:
            st.caption('Enter 1-sigma standard uncertainties. The propagation assumes independent inputs and excludes calibration/model discrepancy.')
            ups=st.number_input('Scattered-power uncertainty (W)',min_value=0.0,value=2.0e-16,format='%.6e',key='tis_thomson_u_scattered')
            upi=st.number_input('Incident-power uncertainty (W)',min_value=0.0,value=0.02,format='%.6e',key='tis_thomson_u_incident')
            ulength=st.number_input('Interaction-length uncertainty (m)',min_value=0.0,value=0.001,format='%.6f',key='tis_thomson_u_length')
            uangle_deg=st.number_input('Scattering-angle uncertainty (degrees)',min_value=0.0,value=0.5,format='%.3f',key='tis_thomson_u_angle')
            usolid=st.number_input('Collection-solid-angle uncertainty (sr)',min_value=0.0,value=2.0e-5,format='%.6e',key='tis_thomson_u_solid')
            uefficiency=st.number_input('Optical-efficiency uncertainty',min_value=0.0,value=0.01,format='%.6f',key='tis_thomson_u_efficiency')
            power_uncertainty_args={
                'scattered_power_uncertainty_W':ups,
                'incident_power_uncertainty_W':upi,
                'interaction_length_uncertainty_m':ulength,
                'scattering_angle_uncertainty_rad':float(np.deg2rad(uangle_deg)),
                'collection_solid_angle_uncertainty_sr':usolid,
                'optical_efficiency_uncertainty':uefficiency,
            }
        if st.button('Estimate Thomson electron density',type='primary',key='run_thomson_power_density'):
            try:
                density_result=density_from_thomson_scattered_power(
                    scattered_power_W=scattered_power,
                    incident_power_W=incident_power,
                    interaction_length_m=interaction_length,
                    scattering_angle_rad=float(np.deg2rad(angle_deg)),
                    collection_solid_angle_sr=solid_angle,
                    optical_efficiency=efficiency,
                    **power_uncertainty_args,
                )
                density_value=float(np.asarray(density_result['electron_density_m3']).item())
                st.success('Calibrated scattered-power density estimate completed.')
                st.metric('Electron density',f'{density_value:.4g} m⁻³')
                if 'electron_density_uncertainty_m3' in density_result:
                    uncertainty_value=float(np.asarray(density_result['electron_density_uncertainty_m3']).item())
                    st.metric('Propagated 1-sigma uncertainty',f'± {uncertainty_value:.4g} m⁻³')
                summary={}
                for key,value in density_result.items():
                    if isinstance(value,np.ndarray) and value.ndim==0:
                        summary[key]=value.item()
                    else:
                        summary[key]=value
                st.json(summary)
                json_download('Export Thomson power-density estimate',summary,'tis_thomson_power_density.json')
            except Exception as e:
                st.error(str(e))
    elif workflow=='Registered analysis modules':
        selected_id=st.selectbox('Analysis',options=[x['id'] for x in ANALYSES],format_func=lambda i: f"{i}. {next(x['name'] for x in ANALYSES if x['id']==i)}")
        entry=next(x for x in ANALYSES if x['id']==selected_id)
        fn=resolve_analysis(selected_id)
        import inspect
        st.caption(f"Callable: tokamak_analysis.{entry['module']}.{entry['function']}")
        contract = entry.get('execution_contract', {})
        completeness = contract.get('completeness', 'incomplete')
        status = ('Implemented within declared scope'
                  if completeness in {'complete', 'complete_with_declared_scope'}
                  else 'Partial implementation / limited scope')
        st.info(f"Execution status: {status}")
        with st.expander('Scientific contract: inputs, outputs and limitations', expanded=(selected_id == 1)):
            st.write(contract.get('algorithm', 'No algorithm description declared.'))
            if contract.get('input_specs'):
                st.markdown('**Required and optional inputs**')
                st.dataframe(pd.DataFrame(contract['input_specs']), use_container_width=True, hide_index=True)
            if contract.get('output_specs'):
                st.markdown('**Outputs**')
                st.dataframe(pd.DataFrame(contract['output_specs']), use_container_width=True, hide_index=True)
            if contract.get('workflow_modes'):
                st.markdown('**Available execution paths**')
                st.dataframe(pd.DataFrame(contract['workflow_modes']), use_container_width=True, hide_index=True)
            if contract.get('assumptions'):
                st.markdown('**Assumptions**')
                for item in contract['assumptions']: st.markdown(f'- {item}')
            if contract.get('limitations'):
                st.markdown('**Scientific limitations**')
                for item in contract['limitations']: st.warning(item)
            st.caption('Implementation maturity describes declared software scope only; it is not a claim about device-specific suitability. Community use and reproducible feedback can improve these modules over time.')
        st.code(str(inspect.signature(fn)),language='python')
        st.write('Enter keyword arguments as JSON. To use an uploaded numeric column, set a value to {"column": "exact column name"}. Other values can be numbers, booleans, strings, lists, or nested arrays as required by the selected function.')
        arg_text=st.text_area('Function arguments (JSON)',value='{}',height=180,key='registered_analysis_args')
        if st.button('Run registered analysis',key='run_registered_analysis'):
            try:
                args=json.loads(arg_text)
                if not isinstance(args,dict): raise ValueError('Arguments must be a JSON object keyed by function parameter name.')
                def resolve_value(value):
                    if isinstance(value,dict) and set(value)=={'column'}:
                        col=value['column']
                        if df is None: raise ValueError('Upload a data file before referencing columns.')
                        if col not in df.columns: raise ValueError(f'Unknown data column: {col}')
                        return pd.to_numeric(df[col],errors='raise').to_numpy(dtype=float)
                    if isinstance(value,list): return [resolve_value(v) for v in value]
                    if isinstance(value,dict): return {k:resolve_value(v) for k,v in value.items()}
                    return value
                result=fn(**{k:resolve_value(v) for k,v in args.items()})
                st.success('Analysis executed.')
                st.write(result)
                if selected_id == 1 and isinstance(result, dict) and {'radius_m', 'density_m3', 'temperature_eV'} <= set(result):
                    profile_frame = pd.DataFrame({'Radius (m)': result['radius_m'], 'Electron density (m^-3)': result['density_m3'], 'Electron temperature (eV)': result['temperature_eV']})
                    st.markdown('**Electron-density profile**')
                    st.line_chart(profile_frame.set_index('Radius (m)')[['Electron density (m^-3)']])
                    st.markdown('**Electron-temperature profile**')
                    st.line_chart(profile_frame.set_index('Radius (m)')[['Electron temperature (eV)']])
                    if 'density_gradient_m-4' in result:
                        st.markdown('**Radial gradients**')
                        gradients = pd.DataFrame({'Radius (m)': result['radius_m'], 'Density gradient (m^-4)': result['density_gradient_m-4'], 'Temperature gradient (eV/m)': result['temperature_gradient_eV_m-1']})
                        st.dataframe(gradients, use_container_width=True, hide_index=True)
                contribution = build_contribution_bundle(
                    analysis_id=int(entry['id']),
                    analysis_name=str(entry['name']),
                    software_version=str(__version__),
                    parameters=args,
                    result=result,
                    dataset_sha256=sha or None,
                    dataset_name=filename or None,
                    tokamak=str(st.session_state.get('tis_source_manifest', {}).get('tokamak', '') or '') or None,
                    shot_id=shot or st.session_state.get('tis_source_manifest', {}).get('shot'),
                    diagnostic=str(entry['name']),
                    researcher_notes="Review metadata and scientific assumptions before sharing.",
                )
                st.download_button(
                    'Export community contribution bundle (JSON)',
                    data=bundle_json(contribution),
                    file_name=f"tis_contribution_analysis_{selected_id:02d}.json",
                    mime='application/json',
                    key=f"download_contribution_{selected_id}",
                    help='Exports analysis parameters, result and provenance. Raw experimental data are not embedded.',
                )
                # Commit the analysis to session history only after all result
                # processing and contribution-bundle construction has succeeded.
                st.session_state['tis_analysis_history']=(st.session_state.get('tis_analysis_history',[])+[{'name':entry['name'],'created_at_utc':datetime.now(timezone.utc).isoformat(),'parameter_snapshot':args,'result_summary':result,'source_filename':filename or None,'source_sha256':sha or None,'shot_id':shot or None,'source_manifest':st.session_state.get('tis_source_manifest'),'import_parameters':st.session_state.get('tis_import_parameters'),'algorithm_version':__version__,'warnings':([str(result.get('caveat'))] if isinstance(result,dict) and result.get('caveat') else [])}])[-100:]
                st.session_state.pop("tis_project_export_bytes", None)
                st.session_state.pop("tis_project_export_signature", None)
            except Exception as e: st.error(f'Analysis failed: {e}')
    elif workflow=='Calibration & uncertainty':
        st.write('Apply y_cal = a y_raw + b with user-supplied calibration coefficients and uncertainty terms.')
        if df is None: st.info('Load a data file to calibrate a numeric channel.')
        else:
            x=col_select('Raw channel',numeric,'adv_cal_x'); a=st.number_input('Slope a',value=1.0); b=st.number_input('Intercept b',value=0.0); sy=st.number_input('Input standard uncertainty',min_value=0.0,value=0.0); sa=st.number_input('Slope uncertainty',min_value=0.0,value=0.0); sb=st.number_input('Intercept uncertainty',min_value=0.0,value=0.0)
            if st.button('Apply calibration'):
                try:
                    val,u=uncertainty_linear(pd.to_numeric(df[x],errors='coerce'),a,b,sy,sa,sb); out=pd.DataFrame({'raw':df[x],'calibrated':val,'standard_uncertainty':u}); st.dataframe(out.head(30)); download_df('Download calibrated channel',out,'tis_calibrated_channel.csv')
                except Exception as e: st.error(str(e))
    elif workflow=='Profile integration & gradients':
        need_data(); x=col_select('Coordinate',numeric,'adv_int_x'); y=col_select('Profile',numeric,'adv_int_y')
        if st.button('Analyze profile'):
            try:
                r=integrate_profile(df[x],df[y]); g=profile_gradient_analysis(df[x],df[y]); st.json({'integral':r,'profile_descriptors':g}); plot_line(df[x],df[y],x,y,'Calibrated profile'); json_download('Export profile analysis',{'integral':r,'profile_descriptors':g},'tis_profile_analysis.json')
            except Exception as e: st.error(str(e))
    elif workflow=='Fast-ion / neutron energy spectrum':
        need_data(); e=col_select('Energy coordinate (eV)',numeric,'adv_energy'); c=col_select('Counts / distribution',numeric,'adv_counts'); bg=st.selectbox('Background (optional)',['<none>']+numeric,key='adv_bg'); sigma_col=st.selectbox('Net-bin uncertainty σ (optional)',['<none>']+numeric,key='adv_energy_sigma')
        poisson=st.checkbox('Assume raw independent Poisson counts (no background subtraction)',value=False,key='adv_poisson')
        if st.button('Calculate spectrum moments'):
            try:
                sigma=None if sigma_col=='<none>' else df[sigma_col]
                r=spectrum_moments(df[e],df[c],None if bg=='<none>' else df[bg],count_uncertainty=sigma,assume_poisson_counts=poisson)
                st.json(r); json_download('Export energy moments',r,'tis_energy_moments.json')
            except Exception as ex: st.error(str(ex))
    elif workflow=='Triple-probe temperature':
        vd=st.number_input('Measured triple-probe voltage difference (V)',min_value=0.000001,value=10.0); mf=st.number_input('Model correction factor',min_value=0.000001,value=1.0)
        if st.button('Estimate triple-probe Te'):
            try: r=triple_probe_temperature_eV(vd,mf); st.json(r); json_download('Export triple-probe result',r,'tis_triple_probe.json')
            except Exception as e: st.error(str(e))
    else:
        need_data(); chosen=st.multiselect('Select synchronized numeric channels',numeric,default=numeric[:min(6,len(numeric))])
        if len(chosen)>=2:
            mat=df[chosen].apply(pd.to_numeric,errors='coerce').corr()
            st.dataframe(mat.style.format('{:.3f}'))
            st.bar_chart(mat.iloc[0].drop(mat.columns[0],errors="ignore").rename("Pearson r"))
            st.caption(f"Pearson correlation coefficients with reference channel {mat.columns[0]!r}; these are not a spatial profile.")
            download_df('Download correlation matrix',mat.reset_index(names='channel'),'tis_correlation_matrix.csv')

elif page=='Diagnostic Catalog':
    render_catalog_page(st, DIAGNOSTICS, FORMULAS, diagnostic_families, json_download, ANALYSES)

elif page=='Project Manager':
    render_project_page(st, session_state=st.session_state, project_title=project, shot_id=shot,
        analyst=analyst, filename=filename, source_sha256=sha, software_version=__version__)

elif page=='Reports & Export':
    st.title('Reports, Provenance & Export')
    if df is None: st.info('Load a file to create a provenance record.'); st.stop()
    quality={}
    for col in numeric[:100]:
        metrics=signal_quality(pd.to_numeric(df[col],errors="coerce").to_numpy(dtype=float))
        quality[str(col)]={k:v for k,v in metrics.items() if k!="n_total"}
    quality_frame=pd.DataFrame(quality).T if quality else pd.DataFrame()
    st.markdown("### Input data quality summary")
    if not quality_frame.empty: st.dataframe(quality_frame,use_container_width=True)
    history=st.session_state.get("tis_analysis_history",[])
    record={'software':'Tokamak Insight Suite','version':__version__,'project':project,'shot_id':shot,'analyst':analyst,'input_filename':filename,'input_size_bytes':len(raw),'input_sha256':sha,'imported_at_utc':st.session_state.get('tis_imported_at_utc'),'source_manifest':st.session_state.get('tis_source_manifest'),'import_parameters':st.session_state.get('tis_import_parameters'),'analysis_history':history,'data_quality':quality,'rows':int(len(df)),'columns':[str(x) for x in df.columns],'numeric_columns':numeric,'diagnostic_count':len(DIAGNOSTICS),'formula_count':len(FORMULAS),'diagnostic_maturity_counts':maturity_counts(DIAGNOSTICS),'analysis_maturity_counts':maturity_counts(ANALYSES),'analysis_maturity_registry':[{'id':item['id'],'name':item['name'],'maturity_level':maturity_label(item)} for item in ANALYSES],'research_release_model':'Open research beta; independent scientific validation is not a prerequisite for use or contribution.','python_version':sys.version,'platform':platform.platform(),'numpy_version':np.__version__,'pandas_version':pd.__version__,'scipy_version':__import__('scipy').__version__,'status':'Software provenance and data-quality report. Researchers are responsible for confirming units, calibration, geometry, uncertainty and model assumptions for their use case.'}
    st.markdown("### Analysis history")
    if history:
        st.dataframe(pd.DataFrame([{"analysis":h["name"],"created_at_utc":h["created_at_utc"]} for h in history]),use_container_width=True)
    else:
        st.info("No analysis export has been generated in this session yet.")
    st.json(record); json_download('Download provenance JSON',record,'tis_provenance.json'); download_df('Export imported data CSV',df,'tis_imported_data.csv')
    st.markdown('### Human-readable report')
    sections=st.session_state.get('tis_report_sections',[])
    st.caption(f'{len(sections)} result section(s) collected this session (ELM/sawtooth/quench and confinement analyses add themselves; run them first).')
    html_report=build_html_report(_json_safe(record),sections)
    st.download_button('Download HTML report',html_report.encode('utf-8'),'tis_report.html','text/html')
    st.markdown('### Session activity')
    st.write(st.session_state.get('tis_activity',[]))

# Activity tracking, bounded to avoid unbounded session growth.
activity=st.session_state.get('tis_activity',[])
entry=f'{page} opened · {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}'
if not activity or activity[0].split(' · ')[0]!=page: st.session_state['tis_activity']=([entry]+activity)[:10]
st.divider()
st.caption(f'Tokamak Insight Suite v{__version__} · diagnostic reference catalog and analysis workflows. Every output must be interpreted using the actual device calibration, units, geometry and uncertainty model. Specialized equilibrium solvers and collisional-radiative atomic models remain distinct modules, not implied by the catalog.')