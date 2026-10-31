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
from tokamak_analysis.analysis_registry import ANALYSES, ANALYSIS_FAMILIES, resolve_analysis
from tokamak_analysis.molecular_spectroscopy import analyze_molecular_spectrum, fit_custom_molecular_model, fit_molecular_boltzmann_temperature
from tokamak_analysis.cxrs import analyze_cxrs_line
from tokamak_analysis.ion_temperature import ion_temperature_from_doppler_spectrum
from tokamak_analysis.poloidal_field_probes import analyze_poloidal_field_probes
from tokamak_analysis.langmuir_probe import (analyze_langmuir_probe_array, analyze_langmuir_probe_series, fit_langmuir_electron_models, fit_langmuir_stretched_exponential, estimate_multispecies_ion_density, estimate_electron_density_from_saturation, estimate_ion_particle_flux, estimate_smoothed_plasma_potential_candidates, langmuir_fit_window_sensitivity)
from tokamak_analysis.mirnov_analysis import analyze_mirnov_signal, magnetic_perturbation_from_mirnov_voltage, analyze_mirnov_array
from tokamak_analysis.thomson_inference import analyze_thomson_spectra
from tokamak_analysis.thomson_workflow import reconstruct_thomson_profile_from_spectra
from tokamak_analysis.cts_analysis import analyze_cts_spectrum, analyze_cts_spectrum_model
from tokamak_analysis.ece_analysis import analyze_ece, reconstruct_ece_temperature_profile, convolve_ece_receiver_response, brightness_temperature_K
from tokamak_analysis.flux_loops import analyze_flux_loop
from tokamak_analysis.poloidal_flux import reconstruct_poloidal_flux
from tis.catalog import DIAGNOSTICS, FORMULAS, diagnostic_families, maturity_counts, maturity_label
from tis.signals import sampling_audit, hampel_despike, prepare_time_series_robust
from tis.events import detect_elms, detect_sawteeth, current_quench_analysis
from tis.plasma import (confinement_summary, beta_metrics, collisionality_nu_star, spitzer_resistivity,
    larmor_radii_and_speeds, coulomb_logarithm_electron, shaping_surface_area, q95_iter_formula)
from tis.report import build_html_report, figure_to_png_base64
from tis.contributions import build_contribution_bundle, bundle_json
from tis.data_sources import fetch_fair_mast_signal, fetch_fair_mast_efit_snapshot, fetch_public_efit_flux_map, list_fair_mast_signals, fetch_public_table_bytes
from tis.core import (load_table, signal_quality, sha256_bytes,
    prepare_time_series, butterworth_filter, fft_spectrum, welch_psd, cross_analysis, cross_correlation_lag,
    fit_spectral_line, doppler_temperature_eV, doppler_temperature_with_uncertainty_eV, doppler_velocity_with_uncertainty_m_s, doppler_velocity_m_s, derive_parameters,
    uncertainty_linear, grid_from_table, regularized_tomography, select_tikhonov_lambda, doppler_temperature_from_fwhm_eV, probe_iv_summary,
    linear_phase_density, langmuir_iv_analysis, suggest_langmuir_fit_windows, mode_number_from_phase, mode_number_from_channels, fluctuation_flux,
    exponential_heat_flux_fit, eich_heat_flux_fit, integrate_profile, spectrum_moments, neutron_rate_from_counts,
    equilibrium_grid_derivatives, xray_spectrum_summary, abel_reconstruction, axisymmetric_radiated_power, profile_gradient_analysis, triple_probe_temperature_eV, integrate_time_series, reflectometry_cutoff_density, electron_cyclotron_resonance_field, corrected_line_ratio, time_frequency_spectrogram, autocorrelation_analysis)

from tis.core import (load_table, signal_quality, sha256_bytes,
    prepare_time_series, butterworth_filter, fft_spectrum, welch_psd, cross_analysis, cross_correlation_lag,
    fit_spectral_line, doppler_temperature_eV, doppler_temperature_with_uncertainty_eV, doppler_velocity_with_uncertainty_m_s, doppler_velocity_m_s, derive_parameters,
    uncertainty_linear, grid_from_table, regularized_tomography, select_tikhonov_lambda, doppler_temperature_from_fwhm_eV, probe_iv_summary,
    linear_phase_density, langmuir_iv_analysis, suggest_langmuir_fit_windows, mode_number_from_phase, mode_number_from_channels, fluctuation_flux,
    exponential_heat_flux_fit, eich_heat_flux_fit, integrate_profile, spectrum_moments, neutron_rate_from_counts,
    equilibrium_grid_derivatives, xray_spectrum_summary, abel_reconstruction, axisymmetric_radiated_power, profile_gradient_analysis, triple_probe_temperature_eV, integrate_time_series, reflectometry_cutoff_density, electron_cyclotron_resonance_field, corrected_line_ratio, time_frequency_spectrogram, autocorrelation_analysis)

def _parse_ts_vector(raw, label):
    """Parse a real three-component vector for Thomson/CTS UI workflows."""
    try:
        values = np.asarray([float(value.strip()) for value in str(raw).split(",") if value.strip()], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be three comma-separated numeric components.") from exc
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain exactly three finite components.")
    return values


def _parse_ts_complex_vector(raw, label):
    """Parse a complex polarization vector from JSON or comma-separated components.

    JSON entries may be real numbers or strings such as "1+0.2j".
    Plain comma-separated real components remain supported.
    """
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("[") else [
            part.strip() for part in str(raw).split(",") if part.strip()
        ]
        values = np.asarray([complex(value) for value in parsed], dtype=complex)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be three comma-separated values or a JSON array of three real/complex components.") from exc
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain exactly three finite complex components.")
    if np.linalg.norm(values) == 0:
        raise ValueError(f"{label} must be non-zero.")
    return values


def _parse_ts_float_list(raw, label):
    try:
        values = np.asarray([float(value.strip()) for value in str(raw).split(",") if value.strip()], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be comma-separated numeric values.") from exc
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain finite numeric values.")
    return values


def _parse_ts_scalar(raw, label, minimum=None, maximum=None):
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be supplied as a numeric value.") from exc
    if not np.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{label} is outside its permitted finite range.")
    return value


def _parse_ts_optional_scalar(raw, label):
    return None if raw is None or not str(raw).strip() else _parse_ts_scalar(raw, label, minimum=0.0)

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
        st.button('Clear loaded data',width="stretch",on_click=_clear_loaded_data)
    st.divider()
    pages=['Home','Public Data Access','Data Import & QC','Time Series & MHD','Events & Transients','Spectral Analysis','Probe Analysis','Plasma Parameters','Confinement & Scalings','Equilibrium & Reconstruction','Tomography & Bolometry','Advanced Analysis','Diagnostic Catalog','Project Manager','Reports & Export']
    requested=st.session_state.pop('jump',None)
    if requested in pages:
        # Apply navigation before the keyed radio widget is instantiated. This
        # also overrides an existing workspace_page value during reruns.
        st.session_state['workspace_page']=requested
    page=st.radio('WORKSPACE',pages,index=0,key='workspace_page',label_visibility='collapsed')
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
    st.markdown('From supplied diagnostic data to physical insight — import, inspect, analyze, reconstruct and export results in one Python-based workspace.')
    a,b,c=st.columns([1.2,1,1])
    with a:
        st.markdown('### Project workspace')
        st.write(f'**Project:** {project}')
        st.write(f'**Shot / discharge:** {shot or "Not specified"}')
        st.write(f'**Input:** {filename or "No data loaded"}')
        st.write('**Data integrity:** '+(sha[:16]+'…' if sha else 'Not available'))
        if st.button('Open Data Import & QC',width="stretch"): st.session_state['jump']='Data Import & QC'; st.rerun()
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
            st.dataframe(df.head(8),width="stretch")
        else:
            st.info('Upload a CSV, TXT, Excel or Parquet file from the left panel to activate quick preview and analysis tools.')
            st.markdown('#### Quick tools')
            st.write('📂 Load Data  ·  〰️ Quick Analysis  ·  🧮 Plasma Parameters  ·  📚 Diagnostic Catalog')
    with right:
        st.markdown('### Quick tools')
        for label,dest in [('Load Data','Data Import & QC'),('Quick Analysis','Time Series & MHD'),('Calculate Plasma Parameters','Plasma Parameters'),('Open Diagnostic Catalog','Diagnostic Catalog')]:
            if st.button(label,width="stretch",key='quick_'+dest): st.session_state['jump']=dest; st.rerun()
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
        st.dataframe(inventory_frame, width="stretch", hide_index=True)
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
        st.dataframe(st.session_state['tis_df'].head(30), width="stretch")
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
        st.dataframe(efit_frame.head(12), width="stretch")
        st.download_button('Download EFIT grid CSV', data=efit_frame.to_csv(index=False).encode('utf-8'), file_name=f"mast_{efit_manifest['shot']}_efit_grid.csv", mime='text/csv')
        st.download_button('Download EFIT manifest', data=json.dumps(efit_manifest, indent=2), file_name='mast_efit_manifest.json', mime='application/json')
        st.caption(efit_manifest['scientific_limit'])

elif page=='Data Import & QC':
    st.title('Data Import & Quality Control')
    if df is None: st.info('Use the upload control in the left panel. Raw input is preserved and never overwritten.'); st.stop()
    st.success(f'{filename}: {len(df):,} rows × {len(df.columns)} columns')
    m1,m2,m3,m4=st.columns(4); m1.metric('Rows',f'{len(df):,}'); m2.metric('Columns',len(df.columns)); m3.metric('Numeric columns',len(numeric)); m4.metric('SHA-256',sha[:12]+'…')
    st.dataframe(df.head(50),width="stretch")
    st.markdown('### Quality audit')
    qcol=col_select('Signal / column',numeric,'qc_col')
    q=signal_quality(pd.to_numeric(df[qcol],errors='coerce').to_numpy()); st.json(q)
    missing=pd.DataFrame({'column':df.columns.astype(str),'dtype':[str(t) for t in df.dtypes],'missing_count':df.isna().sum().values,'missing_fraction':df.isna().mean().values})
    st.dataframe(missing,width="stretch")
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
    st.markdown('### Rogowski coil: calibrated current reconstruction')
    st.caption('Input must already be calibrated to dIp/dt in A/s or plasma current in A. TIS does not apply calibration gains or offsets.')
    acquisition_mode=st.radio('Input signal representation',['Calibrated dIp/dt (A/s; integrate once)','Calibrated plasma current (A; no integration)'],key='rogowski_acquisition_mode')
    int_initial=st.number_input('Initial plasma current for dIp/dt integration (A)',value=0.0,format='%.8g',key='int_initial')
    use_baseline=False
    baseline_start=baseline_end=drift_start=drift_end=None
    if acquisition_mode.startswith('Calibrated dIp/dt'):
        use_baseline=st.checkbox('Subtract mean pre-shot baseline from dIp/dt',value=False,key='int_use_baseline')
        if use_baseline:
            tmin=float(np.min(t)); tmax=float(np.max(t))
            b1,b2=st.columns(2)
            with b1: baseline_start=st.number_input('Baseline start (s)',value=tmin,min_value=tmin,max_value=tmax,key='int_baseline_start')
            with b2: baseline_end=st.number_input('Baseline end (s)',value=min(tmin+(tmax-tmin)*0.1,tmax),min_value=tmin,max_value=tmax,key='int_baseline_end')
        if use_baseline and st.checkbox('Also remove linear drift using a post-event quiet window',value=False,key='int_use_drift'):
            d1,d2=st.columns(2)
            with d1: drift_start=st.number_input('Post window start (s)',value=max(tmin,tmax-(tmax-tmin)*0.1),min_value=tmin,max_value=tmax,key='int_drift_start')
            with d2: drift_end=st.number_input('Post window end (s)',value=tmax,min_value=tmin,max_value=tmax,key='int_drift_end')
            st.caption('Drift correction assumes the true derivative is approximately zero in both selected windows.')
    if st.button('Analyze Rogowski / calibrated current trace',key='run_rogowski_analysis'):
        try:
            if acquisition_mode.startswith('Calibrated dIp/dt'):
                result=integrate_time_series(t,y,scale=1.0,initial_value=int_initial,baseline_start_s=baseline_start,baseline_end_s=baseline_end,drift_start_s=drift_start,drift_end_s=drift_end)
                current=np.asarray(result['integrated_signal'],dtype=float)
                corrected=np.asarray(result['calibrated_signal'],dtype=float)
                metadata={k:v for k,v in result.items() if k not in {'time_s','calibrated_signal','integrated_signal'}}
                metadata.update({'diagnostic':'Rogowski coil','input_representation':'calibrated_dIp_dt_A_s','current_unit':'A','integration_method':'cumulative trapezoidal rule'})
            else:
                current=np.asarray(y,dtype=float).copy()
                corrected=current.copy()
                metadata={'diagnostic':'Rogowski coil','input_representation':'calibrated_current_A','current_unit':'A','integration_applied':False,'note':'Input is already calibrated plasma current; no scaling or second integration was applied.'}
            if not np.all(np.isfinite(current)):
                raise ValueError('Current trace contains non-finite values.')
            metadata.update({'n_samples':int(len(t)),'duration_s':float(t[-1]-t[0]),'median_dt_s':float(np.median(np.diff(t))), 'current_min_A':float(np.min(current)), 'current_max_A':float(np.max(current)), 'current_final_A':float(current[-1])})
            st.success('Rogowski-coil analysis completed from calibrated input.')
            st.json(metadata)
            plot_line(t,current,'Time (s)','Plasma current (A)','Rogowski-derived plasma current')
            out=pd.DataFrame({'time_s':t,'input_signal':y,'analysis_signal':corrected,'plasma_current_A':current})
            download_df('Download Rogowski current series',out,'tis_rogowski_current.csv')
            json_download('Export Rogowski analysis manifest',metadata,'tis_rogowski_manifest.json')
            add_report_section('Rogowski coil current reconstruction',metadata)
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
    mode=st.radio('Workflow',['Optical / emission line','Ion temperature from Doppler spectroscopy','CXRS line profile','Thomson multi-channel spectral inference','Molecular spectroscopy','Molecular Boltzmann temperature','ECE channel/profile analysis','Mirnov / pickup coils','Line-ratio analysis','X-ray energy spectrum','Generic energy-spectrum moments'],horizontal=True,key='spectral_analysis_workflow')
    if mode=='Thomson multi-channel spectral inference':
        st.caption('Primary Analysis 1: fit measured, calibrated multi-channel Thomson spectra and infer physical parameters. No detector calibration, data repair, or experimental validation is performed.')
        all_cols=[str(c) for c in df.columns]
        numeric_cols=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols)<3:
            st.warning('The uploaded table needs numeric radius, wavelength and spectral-intensity columns, plus a channel identifier.')
        else:
            with st.expander('Measured spectrum table mapping', expanded=True):
                channel_col=st.selectbox('Spatial channel identifier',all_cols,index=None,placeholder='Select channel-ID column',key='ts_infer_channel')
                radius_col=st.selectbox('Calibrated channel coordinate (m)',numeric_cols,index=None,placeholder='Select radius column',key='ts_infer_radius')
                wavelength_col=st.selectbox('Calibrated wavelength (nm)',numeric_cols,index=None,placeholder='Select wavelength column',key='ts_infer_wavelength')
                signal_col=st.selectbox('Calibrated spectral intensity',numeric_cols,index=None,placeholder='Select intensity column',key='ts_infer_signal')
                uncertainty_col=st.selectbox('Per-sample spectral 1σ uncertainty (optional)',['<none>']+numeric_cols,key='ts_infer_uncertainty')
            model=st.selectbox('Physical spectral model',[
                'non_collective_maxwellian',
                'non_collective_kappa',
                'non_collective_relativistic_maxwell_juttner',
            ],index=None,placeholder='Select a physical model',key='ts_infer_model')
            st.caption('Optical Analysis 1 uses three explicitly selected non-collective forward models. Microwave collective Thomson scattering is a separate registered CTS workflow.')
            g1,g2,g3=st.columns(3)
            with g1:
                laser_nm_text=st.text_input('Laser wavelength (nm; required)',value='',key='ts_infer_laser')
                if model=='non_collective_relativistic_maxwell_juttner':
                    laser_unc_text=None
                    st.caption('Laser-wavelength uncertainty is not yet propagated by the relativistic Maxwell-Juttner fit.')
                else:
                    use_laser_unc=st.checkbox('Supply laser-wavelength uncertainty',value=False,key='ts_infer_use_laser_unc')
                    laser_unc_text=st.text_input('Laser wavelength 1σ uncertainty (nm)',value='',key='ts_infer_laser_unc') if use_laser_unc else None
            with g2:
                angle_rad_text=st.text_input('Scattering angle (rad; required)',value='',key='ts_infer_angle')
                if model=='non_collective_relativistic_maxwell_juttner':
                    angle_unc_text=None
                    st.caption('Scattering-angle uncertainty is not yet propagated by the relativistic Maxwell-Juttner fit.')
                else:
                    use_angle_unc=st.checkbox('Supply scattering-angle uncertainty',value=False,key='ts_infer_use_angle_unc')
                    angle_unc_text=st.text_input('Scattering angle 1σ uncertainty (rad)',value='',key='ts_infer_angle_unc') if use_angle_unc else None
            with g3:
                response_upload=st.file_uploader('Measured instrument-response CSV (kernel or square matrix)',type=['csv','txt'],key='ts_infer_response')
                response_unc_upload=st.file_uploader('Optional response-matrix 1σ CSV',type=['csv','txt'],key='ts_infer_response_unc')
                if model in {'non_collective_maxwellian','non_collective_kappa'}:
                    joint_covariance_upload=st.file_uploader('Optional full cross-channel spectral covariance CSV (channel-major order)',type=['csv','txt'],key='ts_infer_joint_covariance')
                else:
                    joint_covariance_upload=None
                    st.caption('Full cross-channel spectral GLS is currently supported for the Maxwellian and Kappa fits, not the relativistic Maxwell-Juttner fit.')
                identity_response=st.checkbox('Use identity response only if the supplied spectrum is already corrected for the instrument line-spread function',value=False,key='ts_infer_identity_response')
            relativistic_incident_direction_text=None
            relativistic_scattered_direction_text=None
            relativistic_polarization_text=None
            if model=='non_collective_relativistic_maxwell_juttner':
                st.markdown('**Relativistic Thomson geometry**')
                rg1,rg2=st.columns(2)
                with rg1:
                    relativistic_incident_direction_text=st.text_input('Incident beam direction (x,y,z)',key='ts_infer_rel_incident_direction')
                with rg2:
                    relativistic_scattered_direction_text=st.text_input('Scattered beam direction (x,y,z)',key='ts_infer_rel_scattered_direction')
                relativistic_polarization_text=st.text_input('Incident polarization vector (x,y,z)',key='ts_infer_rel_incident_polarization')
                st.caption('The relativistic model uses exact photon-frequency kinematics, the measured incident polarization, and assumes bulk electron flow parallel to the scattering vector.')
            absolute_density=st.checkbox('Also infer absolute electron density from calibrated W/nm intensity and supplied optical geometry',value=False,key='ts_infer_absolute_density')
            density_kwargs={}
            density_covariance_upload=None
            joint_density_covariance_upload=None
            if absolute_density:
                st.markdown('**Absolute Thomson-power geometry**')
                d1,d2,d3=st.columns(3)
                with d1:
                    laser_power_text=st.text_input('Laser power at scattering volume (W; required)',value='',key='ts_infer_laser_power')
                    laser_power_unc_text=st.text_input('Laser-power 1σ uncertainty (W; optional)',value='',key='ts_infer_laser_power_unc')
                    interaction_length_text=st.text_input('Interaction length (m; required)',value='',key='ts_infer_interaction_length')
                    interaction_length_unc_text=st.text_input('Interaction-length 1σ uncertainty (m; optional)',value='',key='ts_infer_interaction_length_unc')
                with d2:
                    solid_angle_text=st.text_input('Collection solid angle (sr; required)',value='',key='ts_infer_solid_angle')
                    solid_angle_unc_text=st.text_input('Solid-angle 1σ uncertainty (sr; optional)',value='',key='ts_infer_solid_angle_unc')
                    efficiency_text=st.text_input('Optical efficiency (0–1; required)',value='',key='ts_infer_efficiency')
                    efficiency_unc_text=st.text_input('Optical-efficiency 1σ uncertainty (optional)',value='',key='ts_infer_efficiency_unc')
                with d3:
                    if model=='non_collective_relativistic_maxwell_juttner':
                        incident_direction_text=relativistic_incident_direction_text
                        scattered_direction_text=relativistic_scattered_direction_text
                        polarization_text=relativistic_polarization_text
                        st.caption('The beam directions and incident polarization entered in the relativistic geometry section are reused for absolute-density inference.')
                    else:
                        polarization_text=st.text_input('Incident polarization vector (x,y,z; required)',value='',key='ts_infer_polarization')
                        incident_direction_text=st.text_input('Incident beam direction (x,y,z; required)',value='',key='ts_infer_abs_incident_direction')
                        scattered_direction_text=st.text_input('Scattered beam direction (x,y,z; required)',value='',key='ts_infer_abs_scattered_direction')
                    polarization_unc_text=st.text_input('Polarization-factor 1σ uncertainty (optional)',value='',key='ts_infer_polarization_unc')
                    density_covariance_upload=st.file_uploader('Optional within-channel 6×6 density-input relative covariance CSV',type=['csv','txt'],key='ts_infer_density_covariance')
                    joint_density_covariance_upload=st.file_uploader('Optional full cross-channel (6N)×(6N) density-input covariance CSV (non-collective only)',type=['csv','txt'],key='ts_infer_joint_density_covariance')
                    st.caption('A density covariance file is a complete uncertainty model for its factors. Do not also enter scalar optical uncertainties or joint spectral covariance. The full cross-channel matrix also cannot be combined with response-matrix uncertainty.')
                density_kwargs={}
            if st.button('Infer Thomson plasma parameters from measured spectra',type='primary',key='ts_infer_run'):
                st.session_state.pop('tis_thomson_spectral_inference',None)
                try:
                    if model is None:
                        raise ValueError('Select the physical Thomson model explicitly before running the analysis.')
                    if any(value is None for value in (channel_col,radius_col,wavelength_col,signal_col)):
                        raise ValueError('Select the channel-ID, radius, wavelength and spectral-intensity columns explicitly.')
                    required_cols=[channel_col,radius_col,wavelength_col,signal_col]
                    if len(set(required_cols))!=4:
                        raise ValueError('Channel ID, radius, wavelength and intensity must use four distinct columns.')
                    if uncertainty_col!='<none>' and uncertainty_col in required_cols:
                        raise ValueError('The uncertainty column must be distinct from channel, radius, wavelength and intensity columns.')
                    work=df[required_cols+([] if uncertainty_col=='<none>' else [uncertainty_col])].copy()
                    for column in [radius_col,wavelength_col,signal_col]+([] if uncertainty_col=='<none>' else [uncertainty_col]):
                        work[column]=pd.to_numeric(work[column],errors='coerce')
                    if work[channel_col].isna().any():
                        raise ValueError('Channel identifiers must be present for every spectrum sample.')
                    numeric_values=work[[radius_col,wavelength_col,signal_col]+([] if uncertainty_col=='<none>' else [uncertainty_col])].to_numpy(dtype=float)
                    if not np.all(np.isfinite(numeric_values)):
                        raise ValueError('The selected calibrated measurements contain missing or non-finite values; no rows are dropped or repaired.')
                    channel_ids=list(pd.unique(work[channel_col]))
                    if not channel_ids:
                        raise ValueError('No spatial channels were found.')
                    wavelength_rows=[]; signal_rows=[]; uncertainty_rows=[]; radii=[]
                    expected_samples=None
                    for channel_id in channel_ids:
                        channel_frame=work.loc[work[channel_col]==channel_id]
                        x_values=channel_frame[wavelength_col].to_numpy(dtype=float)
                        y_values=channel_frame[signal_col].to_numpy(dtype=float)
                        r_values=channel_frame[radius_col].to_numpy(dtype=float)
                        if np.unique(r_values).size!=1:
                            raise ValueError(f'Channel {channel_id!r} must have exactly one calibrated radius value.')
                        if x_values.size<7 or np.any(np.diff(x_values)<=0):
                            raise ValueError(f'Channel {channel_id!r} must contain at least seven samples on a strictly increasing wavelength axis; source order is preserved.')
                        if expected_samples is None: expected_samples=x_values.size
                        elif x_values.size!=expected_samples:
                            raise ValueError('All channels must contain the same number of wavelength samples.')
                        wavelength_rows.append(x_values); signal_rows.append(y_values); radii.append(float(r_values[0]))
                        if uncertainty_col!='<none>':
                            uncertainty_rows.append(channel_frame[uncertainty_col].to_numpy(dtype=float))
                    wavelengths=np.vstack(wavelength_rows)
                    spectra=np.vstack(signal_rows)
                    signal_uncertainty=None if uncertainty_col=='<none>' else np.vstack(uncertainty_rows)
                    if response_upload is not None:
                        response_frame=pd.read_csv(response_upload,header=None)
                        instrument_response=response_frame.to_numpy(dtype=float)
                        if instrument_response.ndim==2 and 1 in instrument_response.shape:
                            instrument_response=instrument_response.reshape(-1)
                    elif identity_response:
                        instrument_response=np.eye(expected_samples)
                    else:
                        raise ValueError('Upload the measured instrument-response kernel/matrix, or explicitly declare that the supplied spectra are already corrected and use the identity response.')
                    response_uncertainty=None
                    if response_unc_upload is not None:
                        response_uncertainty=pd.read_csv(response_unc_upload,header=None).to_numpy(dtype=float)
                    joint_spectral_covariance=None
                    if joint_covariance_upload is not None:
                        joint_spectral_covariance=pd.read_csv(joint_covariance_upload,header=None).to_numpy(dtype=float)
                    laser_nm=_parse_ts_scalar(laser_nm_text,'Laser wavelength',minimum=1e-9)
                    angle_rad=_parse_ts_scalar(angle_rad_text,'Scattering angle',minimum=1e-9,maximum=float(np.pi))
                    laser_unc_value=None if laser_unc_text is None else _parse_ts_scalar(laser_unc_text,'Laser-wavelength uncertainty',minimum=0.0)
                    angle_unc_value=None if angle_unc_text is None else _parse_ts_scalar(angle_unc_text,'Scattering-angle uncertainty',minimum=0.0)
                    kwargs={
                        'laser_wavelength_nm':laser_nm,
                        'scattering_angle_rad':angle_rad,
                        'laser_wavelength_uncertainty_nm':laser_unc_value,
                        'scattering_angle_uncertainty_rad':angle_unc_value,
                        'instrument_response':instrument_response,
                        'signal_uncertainty':signal_uncertainty,
                        'instrument_response_uncertainty':response_uncertainty,
                        'joint_spectral_covariance':joint_spectral_covariance,
                        'model':model,
                    }
                    if density_covariance_upload is not None:
                        kwargs['density_relative_covariance']=pd.read_csv(density_covariance_upload,header=None).to_numpy(dtype=float)
                    if joint_density_covariance_upload is not None:
                        kwargs['joint_density_relative_covariance']=pd.read_csv(joint_density_covariance_upload,header=None).to_numpy(dtype=float)
                    if model=='non_collective_relativistic_maxwell_juttner':
                        kwargs.update({
                            'incident_direction':_parse_ts_vector(relativistic_incident_direction_text,'Incident direction'),
                            'scattered_direction':_parse_ts_vector(relativistic_scattered_direction_text,'Scattered direction'),
                            'incident_polarization':_parse_ts_vector(relativistic_polarization_text,'Incident polarization'),
                        })
                    if absolute_density:
                        kwargs.update({
                            'laser_power_w':_parse_ts_scalar(laser_power_text,'Laser power',minimum=1e-30),
                            'laser_power_uncertainty_W':_parse_ts_optional_scalar(laser_power_unc_text,'Laser-power uncertainty'),
                            'interaction_length_m':_parse_ts_scalar(interaction_length_text,'Interaction length',minimum=1e-12),
                            'interaction_length_uncertainty_m':_parse_ts_optional_scalar(interaction_length_unc_text,'Interaction-length uncertainty'),
                            'collection_solid_angle_sr':_parse_ts_scalar(solid_angle_text,'Collection solid angle',minimum=1e-12),
                            'collection_solid_angle_uncertainty_sr':_parse_ts_optional_scalar(solid_angle_unc_text,'Solid-angle uncertainty'),
                            'optical_efficiency':_parse_ts_scalar(efficiency_text,'Optical efficiency',minimum=1e-12,maximum=1.0),
                            'optical_efficiency_uncertainty':_parse_ts_optional_scalar(efficiency_unc_text,'Optical-efficiency uncertainty'),
                            'polarization_factor_uncertainty':_parse_ts_optional_scalar(polarization_unc_text,'Polarization-factor uncertainty'),
                            'spectral_power_density_calibrated':True,
                        })
                    if absolute_density:
                        kwargs.update({
                            'incident_polarization':_parse_ts_vector(polarization_text,'Incident polarization'),
                            'incident_direction':_parse_ts_vector(incident_direction_text,'Incident direction'),
                            'scattered_direction':_parse_ts_vector(scattered_direction_text,'Scattered direction'),
                        })
                    if st.session_state.pop('tis_thomson_profile_reconstruction_requested',False):
                        result=reconstruct_thomson_profile_from_spectra(wavelengths,spectra,radii,**kwargs)
                    else:
                        result=analyze_thomson_spectra(wavelengths,spectra,radii,**kwargs)
                    st.session_state['tis_thomson_spectral_inference']={
                        'data_sha':sha,'channel_col':channel_col,'radius_col':radius_col,
                        'wavelength_col':wavelength_col,'signal_col':signal_col,
                        'uncertainty_col':uncertainty_col,'model':model,'result':result,
                    }
                except Exception as e:
                    st.error(f'Thomson spectral inference failed: {e}')
            saved_ts=st.session_state.get('tis_thomson_spectral_inference')
            if saved_ts and saved_ts.get('data_sha')==sha and saved_ts.get('model')==model:
                result=saved_ts['result']
                st.success('Thomson spectral inference completed within the selected physical model.')
                profile=pd.DataFrame({
                    'radius_m':result['radius_m'],
                    'electron_temperature_eV':result['electron_temperature_profile_eV'],
                    'electron_temperature_uncertainty_eV':result['electron_temperature_uncertainty_profile_eV'],
                    'electron_density_m3':result['electron_density_profile_m3'],
                    'electron_density_uncertainty_m3':result['electron_density_uncertainty_profile_m3'],
                    'electron_velocity_projection_m_s':result['electron_velocity_projection_profile_m_s'],
                    'electron_velocity_projection_uncertainty_m_s':result['electron_velocity_projection_uncertainty_profile_m_s'],
                })
                if model=='non_collective_kappa':
                    profile['electron_kappa_index']=[channel.get('kappa_index',np.nan) for channel in result['channels']]
                    profile['electron_kappa_index_uncertainty']=[channel.get('kappa_index_uncertainty',np.nan) for channel in result['channels']]
                if result.get('absolute_scattered_power_available'):
                    profile['integrated_scattered_power_W']=[channel.get('integrated_scattered_power_W',np.nan) for channel in result['channels']]
                else:
                    profile['integrated_spectral_signal_input_units_nm']=[channel.get('integrated_spectral_signal_input_units_nm',np.nan) for channel in result['channels']]
                st.dataframe(profile,width="stretch")
                distribution_rows=[]
                for channel in result['channels']:
                    distribution=channel.get('spectral_distribution',{})
                    distribution_rows.append({
                        'channel_index':channel.get('channel_index'),
                        'radius_m':channel.get('radius_m'),
                        'analysis_status':distribution.get('analysis_status'),
                        'spectral_centroid_nm':distribution.get('spectral_centroid_nm'),
                        'observed_sigma_nm':distribution.get('observed_sigma_nm'),
                        'spectral_skewness':distribution.get('spectral_skewness'),
                        'spectral_excess_kurtosis':distribution.get('spectral_excess_kurtosis'),
                        'integrated_signed_spectral_signal':distribution.get('integrated_signed_spectral_signal'),
                    })
                st.markdown('**Thomson spectral-distribution diagnostics**')
                st.dataframe(pd.DataFrame(distribution_rows),width="stretch")
                st.line_chart(profile.set_index('radius_m')[['electron_temperature_eV']])
                if np.any(np.isfinite(result['electron_density_profile_m3'])):
                    st.line_chart(profile.set_index('radius_m')[['electron_density_m3']])
                st.caption('Electron density is reported only when identifiable from the selected physical model and supplied calibrated intensity/geometry. Flow is the projection along the scattering wavevector, not a reconstructed 3-D velocity vector.')
                if 'spatial_profile_reconstruction' in result:
                    st.markdown('**Reconstructed spatial profiles from the measured Thomson spectra**')
                    spatial=result['spatial_profile_reconstruction']
                    for parameter_name, parameter_result in spatial['profiles'].items():
                        st.markdown(f'**{parameter_name}**')
                        if parameter_result.get('status') not in (None,'completed'):
                            st.caption(parameter_result.get('reason',parameter_result.get('status')))
                            continue
                        for method_name, method_result in parameter_result.get('methods',{}).items():
                            if method_result.get('status')!='completed':
                                st.caption(f'{method_name}: {method_result.get("status")}')
                                continue
                            curve=pd.DataFrame({
                                'value':method_result['values'],
                                'gradient_per_m':method_result['gradient_per_m'],
                            },index=parameter_result['reconstructed_radius_m'])
                            st.caption(method_name.replace('_',' ').title())
                            st.line_chart(curve[['value']])
                            if method_result.get('uncertainty_available'):
                                band=pd.DataFrame({
                                    'lower_95_percentile':method_result['lower_95_percentile'],
                                    'upper_95_percentile':method_result['upper_95_percentile'],
                                },index=parameter_result['reconstructed_radius_m'])
                                st.line_chart(band)
                export=_json_safe(result)
                json_download('Export Thomson spectral-inference results',export,'tis_thomson_spectral_inference.json')
                add_report_section('Thomson multi-channel spectral inference',export)
    elif mode=='ECE channel/profile analysis':
        st.caption('ECE tools use already-calibrated radiance or brightness-temperature inputs. TIS never calibrates detector data. Cold resonance and circular-field radius estimates are approximate, not EFIT/ray-tracing results.')
        numeric_cols=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols)<2:
            st.warning('Upload a table containing numeric ECE measurements.')
        else:
            ece_tabs=st.tabs(['Brightness temperature','Resonance mapping','Channel fluctuations / PSD','Te profile reconstruction'])

            def _ece_parse_frequencies(raw_text, expected_count):
                try:
                    values=np.asarray([float(part.strip()) for part in raw_text.split(',') if part.strip()],dtype=float)
                except ValueError as exc:
                    raise ValueError('Enter finite channel center frequencies as comma-separated values in Hz.') from exc
                if values.size!=expected_count:
                    raise ValueError(f'Enter exactly {expected_count} frequencies, one for each selected channel.')
                if np.any(~np.isfinite(values)) or np.any(values<=0) or np.unique(values).size!=values.size:
                    raise ValueError('Channel frequencies must be finite, positive and distinct.')
                return values

            def _ece_order_channels(frequencies, columns):
                order=np.argsort(frequencies)
                if not np.array_equal(order,np.arange(len(order))):
                    st.info('Channels and their frequencies were reordered together into ascending frequency order; data pairing is preserved.')
                return frequencies[order],[columns[i] for i in order]

            def _ece_parse_optional_channel_values(raw_text, channel_count, label, *, strictly_positive=False):
                try:
                    values=np.asarray([float(part.strip()) for part in raw_text.split(',') if part.strip()],dtype=float)
                except ValueError as exc:
                    raise ValueError(f'{label} must be numeric and comma-separated.') from exc
                if values.size not in (1,channel_count):
                    raise ValueError(f'{label} must contain one value or exactly {channel_count} channel values.')
                if np.any(~np.isfinite(values)) or np.any(values<0) or (strictly_positive and np.any(values<=0)):
                    rule='finite and strictly positive' if strictly_positive else 'finite and non-negative'
                    raise ValueError(f'{label} values must be {rule}.')
                return float(values[0]) if values.size==1 else values

            with ece_tabs[0]:
                st.markdown('**Rayleigh–Jeans brightness temperature**')
                st.caption('Select columns containing absolutely calibrated spectral radiance in W m⁻² sr⁻¹ Hz⁻¹. Raw detector counts or arbitrary voltage are not accepted as calibrated radiance.')
                rad_cols=st.multiselect('Calibrated spectral-radiance columns (at least 2)',numeric_cols,key='ece_radiance_cols')
                rad_freq_text=st.text_input('Center frequencies in Hz, in the same order as radiance columns',placeholder='80000000000, 90000000000',key='ece_radiance_frequencies')
                rad_harmonic=st.number_input('Resonance harmonic',min_value=1,value=2,step=1,key='ece_rad_harmonic')
                rad_map=st.checkbox('Also calculate approximate resonance radii',value=False,key='ece_rad_map')
                if rad_map:
                    rm1,rm2=st.columns(2)
                    with rm1: rad_r0=st.number_input('Reference major radius R₀ (m)',min_value=0.001,value=1.0,key='ece_rad_r0')
                    with rm2: rad_b0=st.number_input('Toroidal field B₀ at R₀ (T)',min_value=0.001,value=2.0,key='ece_rad_b0')
                if st.button('Calculate brightness temperature',key='ece_brightness_run',type='primary'):
                    try:
                        if len(rad_cols)<2: raise ValueError('Select at least two calibrated radiance channels.')
                        rad_freq=_ece_parse_frequencies(rad_freq_text,len(rad_cols))
                        rad_freq,rad_cols=_ece_order_channels(rad_freq,rad_cols)
                        radiance=df[rad_cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
                        bad_rows=np.flatnonzero(~np.all(np.isfinite(radiance),axis=1))
                        if bad_rows.size:
                            bad_by_channel={str(col):int(np.count_nonzero(~np.isfinite(pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)))) for col in rad_cols}
                            raise ValueError(f'Radiance contains missing/non-finite values in {bad_rows.size} rows; per-channel invalid counts: {bad_by_channel}; first row indices: {bad_rows[:10].tolist()}. Correct the source data; TIS will not silently drop rows.')
                        if np.any(radiance<0): raise ValueError('Calibrated spectral radiance cannot be negative.')
                        result=analyze_ece(rad_freq,harmonic=int(rad_harmonic),spectral_radiance_W_m2_sr_Hz=radiance,mode='brightness_temperature',**({'major_radius_m':float(rad_r0),'B_axis_T':float(rad_b0)} if rad_map else {}))
                        st.success('Brightness temperature calculated from the supplied calibrated radiance.')
                        st.dataframe(pd.DataFrame({'frequency_Hz':result['frequency_Hz'],'brightness_temperature_K_mean':np.mean(result['brightness_temperature_K'],axis=0),'brightness_temperature_eV_mean':np.mean(result['brightness_temperature_eV'],axis=0)}),width="stretch")
                        st.caption(result['temperature_interpretation'])
                        if 'resonance_radius_m' in result: st.dataframe(pd.DataFrame({'frequency_Hz':result['frequency_Hz'],'resonance_field_T':result['resonance_field_T'],'approximate_resonance_radius_m':result['resonance_radius_m']}),width="stretch")
                        export={k:v for k,v in result.items()}
                        json_download('Export brightness-temperature JSON',export,'tis_ece_brightness_temperature.json')
                        add_report_section('ECE brightness temperature',export)
                    except Exception as e: st.error(f'ECE brightness-temperature analysis failed: {e}')

            with ece_tabs[1]:
                st.markdown('**Cold resonance and optional approximate major-radius mapping**')
                map_text=st.text_input('Channel center frequencies in Hz',placeholder='80000000000, 90000000000, 100000000000',key='ece_map_frequencies')
                map_harmonic=st.number_input('Resonance harmonic',min_value=1,value=2,step=1,key='ece_map_harmonic')
                map_radius=st.checkbox('Map to major radius using the circular toroidal-field approximation',value=True,key='ece_map_radius')
                if map_radius:
                    mr1,mr2=st.columns(2)
                    with mr1: map_r0=st.number_input('Reference major radius R₀ (m)',min_value=0.001,value=1.0,key='ece_map_r0')
                    with mr2: map_b0=st.number_input('Toroidal field B₀ at R₀ (T)',min_value=0.001,value=2.0,key='ece_map_b0')
                if st.button('Calculate ECE resonance mapping',key='ece_mapping_run',type='primary'):
                    try:
                        raw_parts=[p for p in map_text.split(',') if p.strip()]
                        map_freq=_ece_parse_frequencies(map_text,len(raw_parts))
                        map_freq=np.sort(map_freq)
                        result=analyze_ece(map_freq,harmonic=int(map_harmonic),mode='resonance_mapping',**({'major_radius_m':float(map_r0),'B_axis_T':float(map_b0)} if map_radius else {}))
                        frame={'frequency_Hz':result['frequency_Hz'],'resonance_field_T':result['resonance_field_T']}
                        if 'resonance_radius_m' in result: frame['approximate_resonance_radius_m']=result['resonance_radius_m']
                        st.dataframe(pd.DataFrame(frame),width="stretch")
                        st.caption('Cold, non-relativistic mapping only. Radius is a one-dimensional toroidal-field estimate, not a flux-surface coordinate.')
                        json_download('Export resonance mapping JSON',result,'tis_ece_resonance_mapping.json')
                        add_report_section('ECE resonance mapping',result)
                    except Exception as e: st.error(f'ECE resonance mapping failed: {e}')

            with ece_tabs[2]:
                st.markdown('**Multi-channel fluctuations and spectra**')
                st.caption('Use calibrated, synchronized channel signals in their declared units. All channels must be finite and time samples approximately uniformly spaced.')
                ece_time_col=st.selectbox('Time column (s)',numeric_cols,key='ece_fluct_time_col')
                fluct_cols=st.multiselect('Signal columns (at least 2)',[c for c in numeric_cols if c!=ece_time_col],key='ece_fluct_cols')
                fluct_freq_text=st.text_input('Center frequencies in Hz, same order as signal columns',placeholder='80000000000, 90000000000',key='ece_fluct_frequencies')
                fluct_harmonic=st.number_input('Resonance harmonic',min_value=1,value=2,step=1,key='ece_fluct_harmonic')
                psd_confidence=st.number_input('Welch PSD confidence level (0–1)',min_value=0.50,max_value=0.99,value=0.95,step=0.01,format='%.2f',key='ece_psd_confidence')
                use_noise=st.checkbox('Compare RMS with supplied per-channel RMS noise estimate',value=False,key='ece_use_noise')
                noise_text=''
                if use_noise: noise_text=st.text_input('Noise RMS (one value or one per selected channel; signal units)',key='ece_noise_text')
                if st.button('Analyze ECE fluctuations and PSD',key='ece_fluct_run',type='primary'):
                    try:
                        if len(fluct_cols)<2: raise ValueError('Select at least two signal channels.')
                        fluct_freq=_ece_parse_frequencies(fluct_freq_text,len(fluct_cols))
                        fluct_freq,fluct_cols=_ece_order_channels(fluct_freq,fluct_cols)
                        time_values=pd.to_numeric(df[ece_time_col],errors='coerce').to_numpy(dtype=float)
                        signals=df[fluct_cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
                        bad_time=np.flatnonzero(~np.isfinite(time_values))
                        bad_signal=np.flatnonzero(~np.all(np.isfinite(signals),axis=1))
                        if bad_time.size or bad_signal.size:
                            bad_by_channel={str(col):int(np.count_nonzero(~np.isfinite(pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)))) for col in fluct_cols}
                            raise ValueError(f'Invalid input rows: {bad_time.size} rows with invalid time and {bad_signal.size} rows with one or more invalid channel samples. Per-channel invalid counts: {bad_by_channel}. First invalid time rows: {bad_time[:10].tolist()}; signal rows: {bad_signal[:10].tolist()}. Correct the source data; rows are not silently dropped.')
                        if np.any(np.diff(time_values)<=0): raise ValueError('Time must be strictly increasing; sort/fix the source table without breaking channel synchronization.')
                        noise=None
                        if use_noise:
                            try: noise_values=np.asarray([float(p.strip()) for p in noise_text.split(',') if p.strip()],dtype=float)
                            except ValueError as exc: raise ValueError('Noise RMS must be numeric and comma-separated.') from exc
                            if noise_values.size==1: noise=float(noise_values[0])
                            elif noise_values.size==len(fluct_cols): noise=noise_values
                            else: raise ValueError('Enter one noise RMS value or one per channel.')
                        result=analyze_ece(fluct_freq,harmonic=int(fluct_harmonic),signal=signals,time_s=time_values,noise_std=noise,psd_confidence_level=float(psd_confidence),mode='channel_fluctuations')
                        st.success('ECE fluctuation and spectral analysis completed.')
                        st.dataframe(pd.DataFrame({'frequency_Hz':result['frequency_Hz'],'fluctuation_rms':result['fluctuation_rms']}),width="stretch")
                        psd_frame=pd.DataFrame(result['fluctuation_psd'],index=result['fluctuation_psd_frequency_Hz'],columns=[str(c) for c in fluct_cols])
                        psd_frame.index.name='frequency_Hz'
                        st.markdown('**Full-record periodogram**')
                        st.line_chart(psd_frame)
                        welch_frame=pd.DataFrame(result['welch_psd'],index=result['welch_psd_frequency_Hz'],columns=[str(c) for c in fluct_cols])
                        welch_frame.index.name='frequency_Hz'
                        st.markdown('**Welch-averaged PSD (Hann window, 50% overlap)**')
                        st.line_chart(welch_frame)
                        ci_frame=pd.DataFrame(index=result['welch_psd_frequency_Hz'])
                        for channel_index, channel_name in enumerate(fluct_cols):
                            ci_frame[f'{channel_name} lower CI']=result['welch_psd_confidence_lower'][:,channel_index]
                            ci_frame[f'{channel_name} upper CI']=result['welch_psd_confidence_upper'][:,channel_index]
                        ci_frame.index.name='frequency_Hz'
                        with st.expander('Welch PSD confidence limits and diagnostics'):
                            st.caption(f"Approximate {100*result['welch_psd_confidence_level']:.0f}% confidence interval; effective degrees of freedom ≈ {result['welch_psd_degrees_of_freedom_approx']:.3g}, segments = {result['welch_psd_segment_count']}. Bounds account approximately for overlap correlation; interpret cautiously for short or non-stationary records.")
                            st.line_chart(ci_frame)
                            st.dataframe(pd.DataFrame({'frequency_Hz':result['welch_psd_frequency_Hz'],**{f'{c} PSD':result['welch_psd'][:,i] for i,c in enumerate(fluct_cols)},**{f'{c} lower CI':result['welch_psd_confidence_lower'][:,i] for i,c in enumerate(fluct_cols)},**{f'{c} upper CI':result['welch_psd_confidence_upper'][:,i] for i,c in enumerate(fluct_cols)}}),width="stretch")
                            if 'white_noise_psd_expected' in result:
                                noise_frame=pd.DataFrame(result['white_noise_psd_expected'],index=result['welch_psd_frequency_Hz'],columns=[str(c) for c in fluct_cols])
                                noise_frame.index.name='frequency_Hz'
                                st.caption('Expected one-sided PSD for a white-noise process using the supplied time-domain RMS estimate; this is a comparison reference, not noise subtraction.')
                                st.line_chart(noise_frame)
                        st.caption(f"Sampling frequency: {result['sampling_frequency_Hz']:.6g} Hz · Nyquist: {result['nyquist_frequency_Hz']:.6g} Hz · Welch segment: {result['welch_psd_nperseg']} samples · maximum relative sample-interval deviation: {result['sampling_interval_max_deviation_fraction']:.3g}.")
                        json_download('Export ECE fluctuation/PSD JSON',result,'tis_ece_fluctuation_psd.json')
                        add_report_section('ECE fluctuations and PSD',result)
                    except Exception as e: st.error(f'ECE fluctuation analysis failed: {e}')

            with ece_tabs[3]:
                st.markdown('**Radial Te profile from already-calibrated brightness temperature**')
                st.caption('TIS performs no calibration. Input channel values must already be calibrated brightness temperatures in eV. Channel frequencies and selected data columns are sorted together, preserving their pairing.')
                ece_time_col=st.selectbox('Time column (s)',numeric_cols,key='ece_time_col')
                ece_channel_cols=st.multiselect('Calibrated brightness-temperature channels in eV (at least 3)',[c for c in numeric_cols if c!=ece_time_col],key='ece_channel_cols')
                ece_frequency_text=st.text_input('Channel center frequencies in Hz, in the same order as selected channels',placeholder='80000000000, 90000000000, 100000000000',key='ece_frequency_text')
                use_bandwidth=st.checkbox('Include finite channel bandwidth in the radial localization envelope',value=False,key='ece_use_bandwidth')
                bandwidth_text=''
                if use_bandwidth:
                    bandwidth_text=st.text_input('Channel bandwidths in Hz (one value or one per selected channel)',placeholder='1000000000 or 1000000000, 1000000000, 1000000000',key='ece_bandwidth_text')
                with st.expander('Optional measurement and model-input uncertainties'):
                    use_tb_uncertainty=st.checkbox('Provide brightness-temperature uncertainty (eV)',key='ece_use_tb_uncertainty')
                    tb_uncertainty_text=st.text_input('Tb uncertainty (one value or one per selected channel; eV)',key='ece_tb_uncertainty_text') if use_tb_uncertainty else ''
                    use_frequency_uncertainty=st.checkbox('Provide center-frequency uncertainty (Hz)',key='ece_use_frequency_uncertainty')
                    frequency_uncertainty_text=st.text_input('Frequency uncertainty (one value or one per selected channel; Hz)',key='ece_frequency_uncertainty_text') if use_frequency_uncertainty else ''
                    use_tau_uncertainty=st.checkbox('Provide optical-depth uncertainty',key='ece_use_tau_uncertainty')
                    tau_uncertainty_text=st.text_input('Optical-depth uncertainty (one value or one per selected channel)',key='ece_tau_uncertainty_text') if use_tau_uncertainty else ''
                    bg_uncertainty=st.number_input('Background-temperature uncertainty (eV)',min_value=0.0,value=0.0,format='%.6g',key='ece_bg_uncertainty')
                    r0_uncertainty=st.number_input('Reference-radius uncertainty (m)',min_value=0.0,value=0.0,format='%.6g',key='ece_r0_uncertainty')
                    b0_uncertainty=st.number_input('Toroidal-field uncertainty (T)',min_value=0.0,value=0.0,format='%.6g',key='ece_b0_uncertainty')
                include_time_resolved=st.checkbox('Also reconstruct time-resolved Te(R,t) within the selected window',value=False,key='ece_time_resolved')
                e1,e2,e3=st.columns(3)
                with e1: ece_r0=st.number_input('Reference major radius R₀ (m)',min_value=0.001,value=1.0,format='%.6g',key='ece_r0')
                with e2: ece_b0=st.number_input('Toroidal field B₀ at R₀ (T)',min_value=0.001,value=2.0,format='%.6g',key='ece_b0')
                with e3: ece_harmonic=st.number_input('ECE harmonic',min_value=1,value=2,step=1,key='ece_harmonic')
                inference_basis=st.radio('Temperature inference basis',['Assume optically thick (Tb ≈ Te)','Use supplied optical depth (single-slab LTE)'],key='ece_inference_basis')
                ece_tau=None
                ece_background=0.0
                if inference_basis=='Use supplied optical depth (single-slab LTE)':
                    ece_tau_text=st.text_input('Optical depth τ (one value or one per channel)',placeholder='e.g. 1.0 or 1.0, 1.2, 1.4',key='ece_tau_text')
                    ece_background=st.number_input('Background brightness temperature (eV)',min_value=0.0,value=0.0,format='%.6g',key='ece_background')
                ece_window=st.checkbox('Restrict analysis to a time window',value=False,key='ece_window_enabled')
                ece_window_s=None
                if ece_window:
                    ece_t=pd.to_numeric(df[ece_time_col],errors='coerce').to_numpy(dtype=float)
                    ece_finite_t=ece_t[np.isfinite(ece_t)]
                    if ece_finite_t.size<2 or np.min(ece_finite_t)>=np.max(ece_finite_t):
                        st.error('Selected time column needs at least two distinct finite times.')
                    else:
                        ece_w1,ece_w2=st.columns(2)
                        with ece_w1: ece_t_start=st.number_input('Window start (s)',value=float(np.min(ece_finite_t)),key='ece_t_start')
                        with ece_w2: ece_t_end=st.number_input('Window end (s)',value=float(np.max(ece_finite_t)),key='ece_t_end')
                        if ece_t_start>=ece_t_end: st.error('Window start must be strictly earlier than window end.')
                        elif ece_t_start<np.min(ece_finite_t) or ece_t_end>np.max(ece_finite_t): st.error('Window must remain inside the acquired time range.')
                        else: ece_window_s=(ece_t_start,ece_t_end)
                if st.button('Reconstruct ECE Te profile',type='primary',key='ece_analyze'):
                    try:
                        if len(ece_channel_cols)<3: raise ValueError('Select at least three ECE channels.')
                        ece_frequencies=_ece_parse_frequencies(ece_frequency_text,len(ece_channel_cols))
                        ece_frequency_order=np.argsort(ece_frequencies)
                        ece_frequencies,ece_channel_cols=_ece_order_channels(ece_frequencies,ece_channel_cols)
                        ece_time=pd.to_numeric(df[ece_time_col],errors='coerce').to_numpy(dtype=float)
                        ece_tb=df[ece_channel_cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
                        bad_time=np.flatnonzero(~np.isfinite(ece_time))
                        bad_channel=np.flatnonzero(~np.all(np.isfinite(ece_tb),axis=1))
                        negative_channel=np.flatnonzero(np.any(ece_tb<0,axis=1))
                        bad_by_channel={str(col):int(np.count_nonzero(~np.isfinite(pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)))) for col in ece_channel_cols}
                        negative_by_channel={str(col):int(np.count_nonzero(pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)<0)) for col in ece_channel_cols}
                        st.caption(f'Data-quality audit: {bad_time.size} invalid time rows; {bad_channel.size} rows with missing/non-finite channel values; {negative_channel.size} rows with negative brightness-temperature samples.')
                        if bad_time.size or bad_channel.size: raise ValueError(f'Invalid input rows (time: {bad_time[:10].tolist()}, channels: {bad_channel[:10].tolist()}); per-channel invalid counts: {bad_by_channel}. Correct the source data; TIS will not silently discard rows.')
                        if negative_channel.size: raise ValueError(f'Negative brightness-temperature samples at rows {negative_channel[:10].tolist()}; per-channel negative counts: {negative_by_channel}. Correct the calibrated input.')
                        if np.any(np.diff(ece_time)<=0): raise ValueError('Time must be strictly increasing; sort/fix the source data while preserving channel alignment.')
                        ece_tau=None
                        if inference_basis=='Use supplied optical depth (single-slab LTE)':
                            try: tau_values=np.asarray([float(part.strip()) for part in ece_tau_text.split(',') if part.strip()],dtype=float)
                            except ValueError as exc: raise ValueError('Optical depth must be numeric and comma-separated.') from exc
                            if tau_values.size==1: ece_tau=float(tau_values[0])
                            elif tau_values.size==len(ece_channel_cols): ece_tau=tau_values[ece_frequency_order]
                            else: raise ValueError('Optical depth must be one scalar or one value per selected channel.')
                            if np.any(~np.isfinite(tau_values)) or np.any(tau_values<=0): raise ValueError('Optical depth must be finite and strictly positive.')
                        if ece_window and ece_window_s is None: raise ValueError('Fix the invalid time window before running the profile reconstruction.')
                        bandwidth=None
                        if use_bandwidth:
                            bandwidth=_ece_parse_optional_channel_values(bandwidth_text,len(ece_channel_cols),'Channel bandwidth',strictly_positive=True)
                            if isinstance(bandwidth,np.ndarray): bandwidth=bandwidth[ece_frequency_order]
                        tb_uncertainty=None
                        if use_tb_uncertainty:
                            tb_uncertainty=_ece_parse_optional_channel_values(tb_uncertainty_text,len(ece_channel_cols),'Brightness-temperature uncertainty')
                            if isinstance(tb_uncertainty,np.ndarray): tb_uncertainty=tb_uncertainty[ece_frequency_order]
                        frequency_uncertainty=None
                        if use_frequency_uncertainty:
                            frequency_uncertainty=_ece_parse_optional_channel_values(frequency_uncertainty_text,len(ece_channel_cols),'Frequency uncertainty')
                            if isinstance(frequency_uncertainty,np.ndarray): frequency_uncertainty=frequency_uncertainty[ece_frequency_order]
                        tau_uncertainty=None
                        if use_tau_uncertainty:
                            if ece_tau is None: raise ValueError('Optical-depth uncertainty requires the single-slab optical-depth inference basis.')
                            tau_uncertainty=_ece_parse_optional_channel_values(tau_uncertainty_text,len(ece_channel_cols),'Optical-depth uncertainty')
                            if isinstance(tau_uncertainty,np.ndarray): tau_uncertainty=tau_uncertainty[ece_frequency_order]
                        result=reconstruct_ece_temperature_profile(
                            ece_frequencies,ece_tb,ece_time,major_radius_m=float(ece_r0),B_axis_T=float(ece_b0),
                            harmonic=int(ece_harmonic),optical_depth=ece_tau,background_temperature_eV=float(ece_background),
                            assume_optically_thick=inference_basis=='Assume optically thick (Tb ≈ Te)',time_window_s=ece_window_s,
                            channel_bandwidth_Hz=bandwidth,brightness_temperature_uncertainty_eV=tb_uncertainty,
                            frequency_uncertainty_Hz=frequency_uncertainty,optical_depth_uncertainty=tau_uncertainty,
                            background_temperature_uncertainty_eV=float(bg_uncertainty),
                            major_radius_uncertainty_m=float(r0_uncertainty),B_axis_uncertainty_T=float(b0_uncertainty),
                            include_time_resolved_profile=bool(include_time_resolved))
                        st.success('ECE profile reconstructed within the declared approximate model.')
                        profile_frame={'frequency_Hz':result['frequency_Hz'],'resonance_radius_m':result['resonance_radius_m'],'brightness_temperature_eV':result['brightness_temperature_eV'],'electron_temperature_channel_eV':result['electron_temperature_channel_eV']}
                        if result['electron_temperature_channel_uncertainty_eV'] is not None: profile_frame['electron_temperature_channel_uncertainty_eV']=result['electron_temperature_channel_uncertainty_eV']
                        if result['resonance_radius_uncertainty_m'] is not None: profile_frame['resonance_radius_uncertainty_m']=result['resonance_radius_uncertainty_m']
                        if result['resonance_radius_bandwidth_lower_m'] is not None:
                            profile_frame['bandwidth_radius_lower_m']=result['resonance_radius_bandwidth_lower_m']
                            profile_frame['bandwidth_radius_upper_m']=result['resonance_radius_bandwidth_upper_m']
                        st.dataframe(pd.DataFrame(profile_frame),width="stretch")
                        plot_line(result['resonance_radius_m'],result['electron_temperature_channel_eV'],'Resonance major radius (m)','Te (eV)','ECE channel electron-temperature estimate')
                        plot_line(result['profile_radius_m'],result['electron_temperature_profile_eV'],'Major radius R (m)','Te (eV)','ECE reconstructed radial Te profile')
                        if result['electron_temperature_profile_uncertainty_eV'] is not None:
                            st.line_chart(pd.DataFrame({'Te profile (eV)':result['electron_temperature_profile_eV'],'lower 1-sigma':result['electron_temperature_profile_eV']-result['electron_temperature_profile_uncertainty_eV'],'upper 1-sigma':result['electron_temperature_profile_eV']+result['electron_temperature_profile_uncertainty_eV']},index=result['profile_radius_m']))
                        if include_time_resolved:
                            st.markdown('**Time-resolved Te(R,t) estimate**')
                            time_profile=pd.DataFrame(result['electron_temperature_profile_time_eV'],index=result['selected_time_s'],columns=result['profile_radius_m'])
                            time_profile.index.name='time_s'
                            st.line_chart(time_profile)
                        if result['optical_depth_condition_warning']:
                            st.warning(result['optical_depth_condition_message'])
                        st.caption('Approximate circular-field mapping. Channel bandwidth is reported as a radial localization envelope, not a receiver-response convolution. Uncertainties are first-order and exclude model discrepancy/correlations. No EFIT/ray tracing or relativistic/Doppler correction.')
                        json_download('Export ECE profile analysis JSON',result,'tis_ece_profile_analysis.json')
                        add_report_section('ECE profile reconstruction',result)
                    except Exception as e: st.error(f'ECE profile analysis failed: {e}')
    elif mode=='Mirnov / pickup coils':
        st.caption('Analyze calibrated Mirnov voltage versus time: signal statistics, Welch PSD, in-band peak candidates, optional ideal Faraday-law dB/dt and relative perturbation integration. Instrument transfer functions and absolute-field calibration are not inferred.')
        numeric_cols=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols)<2:
            st.warning('Upload a table with a numeric time column and at least one numeric Mirnov voltage column.')
        else:
            tc=st.selectbox('Time column (s)',numeric_cols,key='mirnov_time')
            vc=st.selectbox('Mirnov voltage column (V)',[c for c in numeric_cols if c!=tc],key='mirnov_voltage')
            c1,c2,c3=st.columns(3)
            with c1: detrend_mode=st.selectbox('Detrending',['constant','linear','none'],key='mirnov_detrend')
            with c2: seg=st.number_input('Welch segment length (samples)',min_value=8,max_value=max(8,int(len(df))),value=min(1024,max(8,int(len(df)))),step=1,key='mirnov_nperseg')
            with c3: prominence=st.number_input('Peak threshold fraction',min_value=0.0,max_value=1.0,value=0.01,step=0.01,key='mirnov_prominence')
            use_geometry=st.checkbox('Estimate ideal dB/dt using coil geometry',key='mirnov_geometry')
            if use_geometry:
                g1,g2=st.columns(2)
                with g1: area=st.number_input('Effective coil area (m²)',min_value=1e-15,value=1e-4,format='%.8g',key='mirnov_area')
                with g2: turns=st.number_input('Coil turns',min_value=1,value=10,step=1,key='mirnov_turns')
            integrate=st.checkbox('Integrate voltage to relative magnetic perturbation (requires coil geometry)',key='mirnov_integrate')
            if integrate and not use_geometry: st.warning('Select “Estimate ideal dB/dt using coil geometry” to enable physically scaled integration.')
            if integrate and use_geometry:
                i1,i2,i3=st.columns(3)
                with i1: baseline=st.number_input('Baseline voltage (V)',value=0.0,format='%.8g',key='mirnov_baseline')
                with i2: initial_db=st.number_input('Initial relative δB (T)',value=0.0,format='%.8g',key='mirnov_initial_db')
                with i3: polarity=st.selectbox('Polarity sign',[-1,1],index=0,key='mirnov_polarity')
            use_array_mode = st.checkbox('Estimate toroidal mode number from a multi-channel Mirnov array', key='mirnov_array_mode')
            if use_array_mode:
                array_columns = st.multiselect(
                    'Mirnov array voltage channels (select at least 3; same voltage units)',
                    [c for c in numeric_cols if c != tc],
                    key='mirnov_array_channels',
                )
                angle_text = st.text_input(
                    'Toroidal sensor angles in radians, in the same order as selected channels',
                    value='',
                    key='mirnov_array_angles',
                    help='Enter comma-separated measured angles, e.g. 0, 1.2, 2.4. Do not use equally spaced defaults unless they match the actual coil geometry.',
                )
                array_nperseg = st.number_input(
                    'Array Welch segment length (samples)', min_value=8,
                    max_value=max(8, int(len(df))),
                    value=min(1024, max(8, int(len(df)))),
                    step=1, key='mirnov_array_nperseg',
                )
                array_reference = st.number_input(
                    'Reference channel index (zero-based)', min_value=0, value=0,
                    step=1, key='mirnov_array_reference',
                )
            if st.button('Analyze Mirnov signal',type='primary',key='mirnov_analyze'):
                try:
                    tt=pd.to_numeric(df[tc],errors='coerce').to_numpy(dtype=float)
                    vv=pd.to_numeric(df[vc],errors='coerce').to_numpy(dtype=float)
                    result=analyze_mirnov_signal(tt,vv,detrend=detrend_mode,nperseg=int(seg),peak_prominence_fraction=float(prominence),**({'coil_area_m2':float(area),'turns':int(turns)} if use_geometry else {}))
                    if use_array_mode:
                        if len(array_columns) < 3:
                            raise ValueError('Select at least three distinct Mirnov array voltage channels.')
                        try:
                            array_angles = np.asarray([float(part.strip()) for part in angle_text.split(',') if part.strip()], dtype=float)
                        except ValueError as exc:
                            raise ValueError('Toroidal angles must be comma-separated finite numbers in radians.') from exc
                        if len(array_angles) != len(array_columns):
                            raise ValueError('The number of toroidal angles must equal the number of selected array channels.')
                        if int(array_reference) >= len(array_columns):
                            raise ValueError('Reference channel index must be smaller than the selected channel count.')
                        array_signals = np.vstack([
                            pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
                            for column in array_columns
                        ])
                        array_result = analyze_mirnov_array(
                            tt, array_signals, array_angles,
                            reference_channel=int(array_reference),
                            nperseg=int(array_nperseg),
                        )
                        result['array_mode_analysis'] = {
                            key: value for key, value in array_result.items()
                            if key not in {'frequency_Hz', 'reference_psd_V2_per_Hz'}
                        }
                    st.success('Mirnov signal analysis completed within the declared numerical scope.')
                    if use_array_mode:
                        mode_result = result['array_mode_analysis']
                        st.subheader('Multi-channel phase / toroidal mode candidate')
                        am1, am2, am3 = st.columns(3)
                        am1.metric('Dominant array frequency', f"{mode_result['dominant_frequency_Hz']:.6g} Hz")
                        am2.metric('Toroidal mode candidate', str(mode_result['estimated_toroidal_mode_number']))
                        am3.metric('Phase-fit RMS', f"{mode_result['phase_fit_rms_rad']:.4g} rad")
                        st.caption('The mode number is an alias-prone candidate based on measured coil angles and cross-spectral phase, not an automatic MHD-mode identification.')
                    m1,m2,m3,m4=st.columns(4)
                    m1.metric('Sampling frequency',f"{result['sampling_frequency_Hz']:.6g} Hz")
                    m2.metric('Nyquist frequency',f"{result['nyquist_frequency_Hz']:.6g} Hz")
                    m3.metric('AC RMS',f"{result['ac_rms_voltage_V']:.6g} V")
                    m4.metric('Peak-to-peak',f"{result['peak_to_peak_voltage_V']:.6g} V")
                    st.dataframe(pd.DataFrame({'Frequency (Hz)':result['frequency_Hz'],'PSD (V²/Hz)':result['power_spectral_density_V2_per_Hz']}),width="stretch")
                    if len(result['dominant_peaks_Hz']): st.dataframe(pd.DataFrame({'Peak frequency (Hz)':result['dominant_peaks_Hz'],'Peak PSD (V²/Hz)':result['dominant_peak_psd_V2_per_Hz']}),width="stretch")
                    else: st.info('No local spectral peaks passed the selected in-band threshold.')
                    if integrate and use_geometry:
                        result['relative_magnetic_perturbation']=magnetic_perturbation_from_mirnov_voltage(tt,vv,int(turns),float(area),initial_delta_B_T=float(initial_db),baseline_voltage_V=float(baseline),polarity_sign=int(polarity))
                    st.json(_json_safe({k:v for k,v in result.items() if k not in {'time_s','voltage_V','frequency_Hz','power_spectral_density','power_spectral_density_V2_per_Hz','dominant_peaks_Hz','dominant_peak_psd_V2_per_Hz','approx_Bdot_T_s','approx_Bdot_detrended_T_s'}}))
                    if integrate and use_geometry:
                        br=result['relative_magnetic_perturbation']
                        st.line_chart(pd.DataFrame({'δB (T)':br['delta_B_T']},index=br['time_s']))
                    json_download('Export Mirnov analysis JSON',result,'tis_mirnov_analysis.json')
                    add_report_section('Mirnov / pickup-coil analysis',result)
                except Exception as e: st.error(f'Mirnov analysis failed: {e}')
    elif mode=='Ion temperature from Doppler spectroscopy':
        st.caption('End-to-end Gaussian line fit and instrument-corrected ion temperature. Requires an independently calibrated wavelength axis, identified isolated transition, emitting-ion mass and measured instrument FWHM. This does not remove Stark/Zeeman broadening, blends, opacity or non-thermal broadening.')
        xcol=col_select('Calibrated wavelength (nm)',numeric,'ti_doppler_x')
        ycol=col_select('Emission intensity',numeric,'ti_doppler_y')
        c1,c2=st.columns(2)
        with c1:
            rest=st.number_input('Laboratory rest wavelength (nm)',min_value=1e-9,value=530.0,format='%.10g',key='ti_doppler_rest')
            mass=st.number_input('Emitting-ion mass (amu)',min_value=1e-9,value=12.0,format='%.8g',key='ti_doppler_mass')
            inst=st.number_input('Instrument Gaussian FWHM (nm; independent calibration)',min_value=0.0,value=0.01,format='%.9g',key='ti_doppler_inst')
            assume_inst_negligible=st.checkbox('Assume instrumental broadening is negligible (independently justified)',value=False,key='ti_doppler_inst_negligible')
        with c2:
            inst_unc=st.number_input('Instrument FWHM uncertainty (nm; 1σ)',min_value=0.0,value=0.0,format='%.9g',key='ti_doppler_inst_unc')
            sigma_col=st.selectbox('Per-sample intensity uncertainty σ (optional)',['<none>']+numeric,key='ti_doppler_sigma')
        _signature={'data_sha':sha,'xcol':xcol,'ycol':ycol,'rest':float(rest),'mass':float(mass),'inst':float(inst),'inst_unc':float(inst_unc),'assume_inst_negligible':bool(assume_inst_negligible),'sigma_col':sigma_col}
        if st.button('Calculate ion temperature from Doppler width',type='primary',key='ti_doppler_run'):
            st.session_state.pop('tis_doppler_ti_result',None)
            try:
                if inst == 0 and not assume_inst_negligible:
                    raise ValueError('Enter the independently calibrated instrument FWHM, or explicitly justify the negligible-instrument-width assumption.')
                _spec=pd.DataFrame({'wavelength_nm':pd.to_numeric(df[xcol],errors='coerce'),'intensity':pd.to_numeric(df[ycol],errors='coerce')})
                if sigma_col!='<none>':
                    _spec['uncertainty']=pd.to_numeric(df[sigma_col],errors='coerce')
                    if not np.all(np.isfinite(_spec['uncertainty'])) or np.any(_spec['uncertainty']<=0):
                        raise ValueError('Selected intensity uncertainty must be finite and positive for every sample.')
                _spec=_spec.dropna(subset=['wavelength_nm','intensity']).sort_values('wavelength_nm',kind='stable')
                _kwargs={}
                if sigma_col!='<none>':
                    _kwargs['intensity_uncertainty']=_spec['uncertainty'].to_numpy(dtype=float)
                _result=ion_temperature_from_doppler_spectrum(
                    _spec['wavelength_nm'].to_numpy(dtype=float),_spec['intensity'].to_numpy(dtype=float),
                    rest_wavelength_nm=float(rest),ion_mass_amu=float(mass),
                    instrumental_fwhm_nm=float(inst),
                    instrumental_fwhm_uncertainty_nm=float(inst_unc),**_kwargs)
                _result={k:(v.tolist() if isinstance(v,np.ndarray) else
                            {kk:(vv.tolist() if isinstance(vv,np.ndarray) else vv) for kk,vv in v.items()} if isinstance(v,dict) else v)
                         for k,v in _result.items()}
                st.session_state['tis_doppler_ti_result']={**_signature,'result':_result,
                    'wavelength_nm':_spec['wavelength_nm'].to_list(),'intensity':_spec['intensity'].to_list()}
            except Exception as e:
                st.error(f'Ion temperature withheld: {e}')
        _saved=st.session_state.get('tis_doppler_ti_result')
        if _saved and all(_saved.get(k)==v for k,v in _signature.items()):
            _r=_saved['result']
            m1,m2,m3,m4=st.columns(4)
            m1.metric('Ion temperature',f"{_r['ion_temperature_eV']:.6g} eV")
            m2.metric('1σ uncertainty',f"{_r['ion_temperature_uncertainty_eV']:.3g} eV" if _r['ion_temperature_uncertainty_eV'] is not None else 'Unavailable')
            m3.metric('Observed FWHM',f"{_r['observed_fwhm_nm']:.6g} nm")
            m4.metric('Fit R²',f"{_r['fit_r_squared']:.6g}")
            st.success('Doppler ion-temperature analysis completed within the declared Gaussian isolated-line model.')
            st.caption(_r['caveat'])
            st.json({k:v for k,v in _r.items() if k not in {'fitted_intensity','fit_residuals','fit_covariance'}})
            st.line_chart(pd.DataFrame({'measured':_saved['intensity'],'fitted':_r['fitted_intensity'],'residual':_r['fit_residuals']},index=_saved['wavelength_nm']))
            json_download('Export Doppler ion-temperature result',_r,'tis_doppler_ion_temperature.json')
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
        st.caption('Fit one calibrated, isolated charge-exchange line. Requires independently known rest wavelength, emitting-ion mass and Gaussian instrumental sigma (not FWHM). Reports line-of-sight velocity and a single-Maxwellian Ti estimate only when the fitted FWHM spans at least three median wavelength intervals; it does not infer impurity density or invert multiple views.')
        xcol=col_select('Calibrated wavelength (nm)',numeric,'cxrs_x')
        ycol=col_select('Spectral intensity',numeric,'cxrs_y')
        c1,c2=st.columns(2)
        with c1:
            rest=st.number_input('Known rest wavelength λ₀ (nm)',min_value=0.000001,value=530.0,format='%.9g',key='cxrs_rest')
            rest_unc=st.number_input('Rest-wavelength uncertainty (nm; 1σ)',min_value=0.0,value=0.0,format='%.9g',key='cxrs_rest_unc')
            mass=st.number_input('Emitting-ion mass (amu)',min_value=0.000001,value=12.0,format='%.8g',key='cxrs_mass')
        with c2:
            inst=st.number_input('Instrument Gaussian sigma (nm; not FWHM)',min_value=0.000001,value=0.008,format='%.9g',key='cxrs_inst_sigma')
            inst_unc=st.number_input('Instrument sigma uncertainty (nm; 1σ)',min_value=0.0,value=0.0,format='%.9g',key='cxrs_inst_sigma_unc')
            viewing_angle=st.number_input('Viewing angle metadata (rad; no deprojection)',min_value=0.0,max_value=float(np.pi),value=0.0,format='%.8g',key='cxrs_viewing_angle')
            sigma_col=st.selectbox('Per-sample intensity uncertainty σ (optional)',['<none>']+numeric,key='cxrs_sigma')
        if st.button('Analyze CXRS line', key='cxrs_run'):
            # Never leave a previous successful result visible if a new attempt fails.
            st.session_state.pop('tis_cxrs_result', None)
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
                    instrument_sigma_uncertainty_nm=float(inst_unc),rest_wavelength_uncertainty_nm=float(rest_unc),viewing_angle_rad=float(viewing_angle),**_kwargs)
                _result_dict=_result.to_dict()
                _result_dict['wavelength_nm']=_spec['wavelength_nm'].to_numpy(dtype=float).tolist()
                _result_dict['measured_signal']=_spec['signal'].to_numpy(dtype=float).tolist()
                st.session_state['tis_cxrs_result']={'data_sha':sha,'xcol':xcol,'ycol':ycol,'rest':float(rest),'rest_unc':float(rest_unc),'mass':float(mass),'inst':float(inst),'inst_unc':float(inst_unc),'viewing_angle':float(viewing_angle),'sigma_col':sigma_col,'result':_result_dict}
            except Exception as e:
                st.error(str(e))
        _saved_cxrs=st.session_state.get('tis_cxrs_result')
        if _saved_cxrs and all(_saved_cxrs.get(k)==v for k,v in {'data_sha':sha,'xcol':xcol,'ycol':ycol,'rest':float(rest),'rest_unc':float(rest_unc),'mass':float(mass),'inst':float(inst),'inst_unc':float(inst_unc),'viewing_angle':float(viewing_angle),'sigma_col':sigma_col}.items()):
            _rr=_saved_cxrs['result']
            st.success('CXRS line-profile analysis completed within the declared single-line scope.')
            _m1, _m2, _m3, _m4 = st.columns(4)
            _m1.metric('Ion temperature', f"{_rr['ion_temperature_eV']:.6g} ± {_rr['ion_temperature_uncertainty_eV']:.3g} eV")
            _m2.metric('Line-of-sight velocity', f"{_rr['line_of_sight_velocity_m_s']:.6g} ± {_rr['velocity_uncertainty_m_s']:.3g} m/s")
            _m3.metric('Fit R²', f"{_rr['fit_r_squared']:.6f}")
            _m4.metric('Reduced χ²', f"{_rr['reduced_chi_square']:.5g}" if _rr['reduced_chi_square'] is not None else "Not calculated (no σ)")
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
        molecular_signature = {
            'data_sha': sha, 'xcol': xcol, 'ycol': ycol,
            'prominence_fraction': float(prominence),
            'baseline_quantile': float(baseline_q),
            'reference_text': refs_text.strip(),
            'ranges_text': ranges_text.strip(),
        }
        if st.button('Analyze molecular spectrum'):
            st.session_state.pop('tis_molecular_spectrum_result', None)
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
                    **molecular_signature, 'result':result
                }
                st.success('Molecular spectrum feature analysis completed.')
            except Exception as e:
                st.error(str(e))
        saved_molecular=st.session_state.get('tis_molecular_spectrum_result')
        if saved_molecular and all(saved_molecular.get(k)==v for k,v in molecular_signature.items()):
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
        st.caption('Input must already be calibrated poloidal magnetic field in tesla. TIS performs no gain/offset calibration.')
        time_col=col_select('Time column (s)',numeric,'pf_probe_time')
        pf_channels=st.multiselect('Poloidal-field probe Bp channel(s) (T)',[c for c in numeric if c != time_col],default=[],key='pf_probe_channels')
        st.caption('Optional uncertainty is a 1-sigma uncertainty in tesla, one value or one value per channel.')
        pf_unc_text=st.text_input('Field uncertainty 1σ (T; optional)','',key='pf_uncertainty')
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
                uncertainty_value=None
                if pf_unc_text.strip():
                    values=[float(part.strip()) for part in pf_unc_text.split(',') if part.strip()]
                    if len(values)==1: values=values*len(pf_channels)
                    if len(values)!=len(pf_channels) or not np.all(np.isfinite(values)) or np.any(np.asarray(values)<0):
                        raise ValueError(f'Field uncertainty must be one non-negative finite value or exactly {len(pf_channels)} values.')
                    uncertainty_value=values[0] if len(pf_channels)==1 else values
                time_values=pd.to_numeric(df[time_col],errors='coerce').to_numpy(dtype=float)
                signal_values=df[pf_channels].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
                result=analyze_poloidal_field_probes(time_values,signal_values,signal_uncertainty=uncertainty_value,baseline_interval_s=baseline_interval,probe_names=pf_channels)
                st.success('Poloidal-field probe analysis completed from calibrated input.')
                st.json(result)
                for channel_index,channel_name in enumerate(pf_channels):
                    st.markdown(f'**{channel_name}**')
                    plot_line(result['time_s'],result['B_poloidal_T'][:,channel_index],time_col,'Bp (T)',f'{channel_name}: poloidal field')
                    plot_line(result['time_s'],result['dB_poloidal_dt_T_s'][:,channel_index],time_col,'dBp/dt (T/s)',f'{channel_name}: time derivative')
                json_download('Export poloidal-field analysis JSON',result,'tis_poloidal_field_probe_analysis.json')
            except Exception as e:
                st.error(f'Poloidal-field probe analysis failed: {e}')
        st.info('Interpretation limits: probe transfer function, bandwidth, geometry, timing uncertainty and correlated drift are not automatically inferred. Numerical differentiation can amplify noise; sampling diagnostics alone do not rule out aliasing.')
    st.divider()
    st.markdown('### Langmuir probe I–V analysis')
    vcol=col_select('Probe voltage (V)',numeric,'probe_v')
    current_options=[c for c in numeric if c != vcol]
    if not current_options:
        st.error('Langmuir analysis requires separate numeric voltage and current columns.')
        st.stop()
    default_current=next((c for c in current_options if str(c).casefold() in {'current_a','current','probe_current_a','i_a'}),current_options[0])
    if st.session_state.get('probe_i') not in current_options:
        st.session_state['probe_i']=default_current
    icol=col_select('Probe current (A)',current_options,'probe_i',default=default_current)
    voltage_values=pd.to_numeric(df[vcol],errors='coerce').to_numpy(dtype=float)
    current_values=pd.to_numeric(df[icol],errors='coerce').to_numpy(dtype=float)
    finite_pairs=np.isfinite(voltage_values)&np.isfinite(current_values)
    paired_voltage=voltage_values[finite_pairs]
    distinct_paired_voltage=np.unique(paired_voltage)
    if np.count_nonzero(finite_pairs) < 8 or distinct_paired_voltage.size < 8:
        st.error('The selected columns need at least eight finite paired samples and eight distinct voltage values among those pairs before Langmuir analysis can run.')
        st.stop()
    vmin=float(np.min(paired_voltage)); vmax=float(np.max(paired_voltage))
    st.markdown('### Fit-window selection')
    st.caption('Fit-window suggestions scan the measured I–V curve for candidate electron-exponential and low-voltage ion-tail intervals. They do not validate physical regimes; inspect the curve, saturation regions and residuals.')
    if st.button('Suggest I–V fit windows', key='probe_suggest_fit_windows'):
        try:
            suggestions=suggest_langmuir_fit_windows(df[vcol],df[icol],current_convention=st.session_state.get('probe_current_convention','electron_positive'))
            st.session_state['probe_fit_window_suggestions']=suggestions
            electron_candidate=suggestions.get('electron_fit_range_V')
            ion_candidate=suggestions.get('ion_fit_range_candidate_V')
            if electron_candidate is not None:
                st.session_state['probe_electron_fit_range']=electron_candidate
                st.session_state['probe_enable_electron_fit']=True
            if ion_candidate is not None:
                st.session_state['probe_ion_fit_range']=ion_candidate
                st.session_state['probe_enable_ion_fit']=True
        except Exception as exc:
            st.session_state['probe_fit_window_suggestions']={'error':str(exc)}
    suggestions=st.session_state.get('probe_fit_window_suggestions')
    if suggestions:
        if suggestions.get('error'):
            st.warning('Fit-window suggestion failed: '+str(suggestions['error']))
        else:
            st.info('Suggested electron window: '+str(suggestions.get('electron_fit_range_V'))+' V; log-fit R²: '+str(suggestions.get('electron_candidate_log_R2'))+'; '+str(suggestions.get('electron_candidate_status'))+'. Ion window: '+str(suggestions.get('ion_fit_range_candidate_V'))+' V ('+str(suggestions.get('ion_candidate_status'))+'). Review both intervals against the measured curve before accepting them.')
    efit=None
    if st.checkbox('Enable electron-retarding exponential fit (Te)',value=False,key='probe_enable_electron_fit'):
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin>=vmax:
            st.error('The selected voltage column must contain a finite, nonzero voltage range.')
        else:
            efit=st.slider('Electron-retarding fit range (V)',min_value=float(vmin),max_value=float(vmax),value=(float(vmin+(vmax-vmin)*.35),float(vmin+(vmax-vmin)*.75)),key='probe_electron_fit_range')
    ifit=None
    if st.checkbox('Enable ion-branch fit (ion-branch slope and intercept)',value=False,key='probe_enable_ion_fit'):
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin>=vmax:
            st.error('The selected voltage column must contain a finite, nonzero voltage range.')
        else:
            ifit=st.slider('Ion-branch fit range (V)',min_value=float(vmin),max_value=float(vmax),value=(float(vmin),float(vmin+(vmax-vmin)*.2)),key='probe_ion_fit_range')
    cur_conv=st.radio('Current sign convention',['electron_positive','ion_positive'],horizontal=True,key='probe_current_convention',
        help='All calculations normalize internally to electron_positive; advanced fits use the same convention.')
    sigma_current=st.number_input('Current standard uncertainty per input sample (A; 0 = not supplied)',min_value=0.0,value=0.0,format='%.6g',key='probe_current_sigma')
    with st.expander('Numerical fit-screen settings (not physical-regime criteria)'):
        threshold_c1,threshold_c2,threshold_c3=st.columns(3)
        with threshold_c1: fit_min_r2=st.number_input('Minimum fit R²',min_value=0.0,max_value=1.0,value=0.98,step=0.01,key='probe_fit_min_r2')
        with threshold_c2: fit_max_nrmse=st.number_input('Maximum normalized RMSE',min_value=1e-6,value=0.1,step=0.01,format='%.4g',key='probe_fit_max_nrmse')
        with threshold_c3: ion_max_relative_change=st.number_input('Maximum ion-branch relative change',min_value=0.0,value=0.1,step=0.01,format='%.4g',key='probe_ion_flatness_threshold')
        st.caption('These are configurable numerical review screens. They are not universal physical acceptance thresholds and cannot certify an ion-saturation or electron-retarding regime.')
    density_enabled=st.checkbox('Enable model-based ion-density estimate',value=False,key='probe_enable_density')
    density_options={}
    if density_enabled:
        with st.container(border=True):
            st.caption('Density inversion uses the published sheath/Bohm current-balance relation. Supply the effective collection area, ion mass, coefficient and model label; no defaults are inferred.')
            dc1,dc2,dc3=st.columns(3)
            with dc1: density_area=st.number_input('Effective collection area (m²)',min_value=1e-20,value=1e-4,format='%.6g',key='probe_density_area')
            with dc2: density_mass=st.number_input('Assumed ion mass (kg)',min_value=1e-30,value=3.34e-27,format='%.6g',key='probe_density_mass')
            with dc3: density_coeff=st.number_input('Sheath/Bohm coefficient',min_value=1e-6,value=0.61,format='%.6g',key='probe_density_coeff')
            density_model=st.text_input('Sheath model/source label',value='',key='probe_density_model')
            density_charge=st.number_input('Ion charge state Zᵢ (positive integer)',min_value=1,value=1,step=1,key='probe_density_charge')
            density_allow_screen_override=st.checkbox(
                'Allow provisional density calculation if a numerical review screen fails',
                value=False,key='probe_density_screen_override',
                help='This bypasses only the configurable numerical review gate. It does not establish that the selected branches are physically valid or in saturation.')
            uc1,uc2,uc3=st.columns(3)
            with uc1: density_sigma_current=st.number_input('Ion saturation-current σ (A; 0 = not supplied)',min_value=0.0,value=0.0,format='%.6g',key='probe_density_sigma_current')
            with uc2: density_area_rel=st.number_input('Area relative σ (0 = not supplied)',min_value=0.0,value=0.0,format='%.6g',key='probe_density_area_rel')
            with uc3: density_mass_rel=st.number_input('Ion-mass relative σ (0 = not supplied)',min_value=0.0,value=0.0,format='%.6g',key='probe_density_mass_rel')
            uc4,uc5,uc6=st.columns(3)
            with uc4: density_coeff_rel=st.number_input('Sheath-coefficient relative σ',min_value=0.0,value=0.0,format='%.6g',key='probe_density_coeff_rel')
            with uc5: density_te_rel=st.number_input('Te relative σ',min_value=0.0,value=0.0,format='%.6g',key='probe_density_te_rel')
            with uc6: density_te_cal_rel=st.number_input('Te calibration relative σ',min_value=0.0,value=0.0,format='%.6g',key='probe_density_te_cal_rel')
            uc7,uc8=st.columns(2)
            with uc7: density_te_window_rel=st.number_input('Te fit-window relative σ',min_value=0.0,value=0.0,format='%.6g',key='probe_density_te_window_rel')
            with uc8: density_model_rel=st.number_input('Density-model relative σ',min_value=0.0,value=0.0,format='%.6g',key='probe_density_model_rel')
            density_options={'probe_area_m2':density_area,'ion_mass_kg':density_mass,
                             'ion_saturation_coefficient':density_coeff,'sheath_model':density_model,
                             'ion_charge_state':int(density_charge),
                             'allow_density_when_numerical_screen_fails':bool(density_allow_screen_override),
                             'ion_current_uncertainty_A':density_sigma_current if density_sigma_current>0 else None,
                             'probe_area_relative_uncertainty':density_area_rel if density_area_rel>0 else None,
                             'ion_mass_relative_uncertainty':density_mass_rel if density_mass_rel>0 else None,
                             'ion_saturation_coefficient_relative_uncertainty':density_coeff_rel if density_coeff_rel>0 else None,
                             'electron_temperature_relative_uncertainty':density_te_rel if density_te_rel>0 else None,
                             'electron_temperature_calibration_relative_uncertainty':density_te_cal_rel if density_te_cal_rel>0 else None,
                             'electron_temperature_fit_window_relative_uncertainty':density_te_window_rel if density_te_window_rel>0 else None,
                             'density_model_relative_uncertainty':density_model_rel if density_model_rel>0 else None}
            st.caption('Only entered uncertainty components are propagated. The fitted Te statistical covariance is required; missing components are not silently set to zero.')
    if st.button('Analyze probe I–V',type='primary'):
        try:
            result=langmuir_iv_analysis(df[vcol],df[icol],efit,ifit,current_convention=cur_conv,
                                        current_uncertainty_A=(sigma_current if sigma_current>0 else None),
                                        fit_quality_min_r2=float(fit_min_r2),
                                        fit_quality_max_nrmse=float(fit_max_nrmse),
                                        ion_flatness_max_relative_change=float(ion_max_relative_change),
                                        **density_options)
            st.subheader('Langmuir analysis results')
            r1,r2=st.columns(2)
            vf=result.get('floating_potential_V')
            te_result=result.get('electron_temperature_eV')
            with r1: st.metric('Floating potential Vf',f'{vf:.5g} V' if vf is not None else 'Not determined')
            with r2: st.metric('Electron temperature Te',f'{te_result:.5g} eV' if te_result is not None else 'Not fitted')
            st.caption('Floating-potential status: '+str(result.get('floating_potential_status','not reported'))+
                       ' · Current convention: '+str(result.get('current_convention',cur_conv)))
            st.write({'Vf standard uncertainty (V)':result.get('floating_potential_standard_uncertainty_V'),
                      'Vf uncertainty status':result.get('floating_potential_uncertainty_status'),
                      'Maximum-dI/dV Vp proxy (V)':result.get('plasma_potential_proxy_V'),
                      'Vp proxy standard uncertainty (V)':result.get('plasma_potential_proxy_standard_uncertainty_V'),
                      'Vp proxy uncertainty status':result.get('plasma_potential_proxy_uncertainty_status'),
                      'd²I/dV² zero-crossing candidates (V)':result.get('plasma_potential_d2IdV2_zero_crossing_candidates_V'),
                      'd²I/dV² candidate standard uncertainties (V)':result.get('plasma_potential_d2IdV2_zero_crossing_standard_uncertainties_V'),
                      'd²I/dV² candidate uncertainty status':result.get('plasma_potential_d2IdV2_zero_crossing_uncertainty_status')})
            quality=result.get('input_quality',{})
            if quality.get('discarded_nonfinite_pair_count',0):
                st.warning(f"Discarded {quality['discarded_nonfinite_pair_count']} paired samples containing non-finite voltage/current values.")
            if quality.get('duplicate_voltage_pair_count_aggregated',0):
                st.info(f"Aggregated {quality['duplicate_voltage_pair_count_aggregated']} duplicate-voltage samples by averaging their currents.")
            if efit is not None:
                st.write({'Electron-fit R²':result.get('electron_retarding_fit_R2'),
                          'Electron-fit normalized RMSE':result.get('electron_retarding_fit_normalized_rmse'),
                          'Electron-fit residual RMS (A)':result.get('electron_retarding_fit_residual_rms_A'),
                          'Te standard uncertainty (eV)':result.get('electron_temperature_standard_error_eV'),
                          'Te uncertainty status':result.get('electron_temperature_uncertainty_status'),
                          'Fit quality screening':result.get('electron_retarding_fit_status')})
            if density_enabled:
                st.write({'Ion density estimate (m⁻³)':result.get('ion_density_estimate_m-3'),
                          'Density status':result.get('ion_density_status'),
                          'Density uncertainty (m⁻³)':result.get('ion_density_standard_uncertainty_m-3'),
                          'Density uncertainty status':result.get('ion_density_uncertainty_status')})
            st.json(result)
            plot_line(result['processed_voltage_V'],result['processed_current_A'],'Probe voltage (V)','Normalized current (A)','Processed calibrated probe I–V')
            json_download('Export probe analysis JSON',result,'tis_probe_analysis.json')
        except Exception as e: st.error(str(e))
    with st.expander('Advanced electron-branch and plasma-potential models'):
        advanced_method=st.selectbox('Advanced analysis',['Single-/two-temperature exponential fit','Stretched-exponential shape fit','Smoothed derivative candidates','Fit-window sensitivity'],key='probe_advanced_method')
        advanced_model='two_temperature' if advanced_method=='Single-/two-temperature exponential fit' else 'stretched' if advanced_method=='Stretched-exponential shape fit' else 'smoothed' if advanced_method=='Smoothed derivative candidates' else 'sensitivity'
        if advanced_method=='Single-/two-temperature exponential fit':
            model_choice=st.selectbox('Electron model',['single_maxwellian','two_temperature'],index=1,key='probe_advanced_electron_model')
            if model_choice=='two_temperature':
                st.caption('Empirical two-exponential decomposition; it does not establish two physical electron populations.')
        if advanced_method=='Stretched-exponential shape fit':
            st.caption('Empirical curve-shape comparison only; the fitted exponent is not an EEDF or kinetic-distribution parameter.')
        needs_electron_window=advanced_method in {'Single-/two-temperature exponential fit','Stretched-exponential shape fit','Fit-window sensitivity'}
        advanced_min_points=(8 if advanced_model=='two_temperature' and model_choice=='two_temperature'
                             else 5 if advanced_model=='stretched' else 4)
        advanced_window_points=(int(np.count_nonzero((distinct_paired_voltage>=efit[0])&(distinct_paired_voltage<=efit[1])))
                                if efit is not None else 0)
        advanced_window_valid=(efit is not None and advanced_window_points>=advanced_min_points)
        if needs_electron_window and not advanced_window_valid:
            st.info(f'Select a valid electron-retarding interval containing at least {advanced_min_points} distinct calibrated voltage values before running this model.')
        if advanced_method=='Smoothed derivative candidates':
            max_sg_window=int(min(101,distinct_paired_voltage.size-(1 if distinct_paired_voltage.size%2==0 else 0)))
            sg1,sg2=st.columns(2)
            with sg1: sg_window=st.number_input('Local-polynomial window (odd distinct samples)',min_value=5,max_value=max_sg_window,value=min(7,max_sg_window),step=2,key='probe_sg_window')
            max_sg_order=min(6,int(sg_window)-1)
            with sg2: sg_order=st.number_input('Polynomial order (at least 2 for d²I/dV²)',min_value=2,max_value=max_sg_order,value=min(2,max_sg_order),key='probe_sg_order')
        if st.button('Run selected advanced analysis',key='probe_run_advanced',disabled=bool(needs_electron_window and not advanced_window_valid)):
            try:
                if advanced_model=='two_temperature':
                    advanced_result=fit_langmuir_electron_models(df[vcol],df[icol],efit,model=model_choice,current_convention=cur_conv)
                elif advanced_model=='stretched':
                    advanced_result=fit_langmuir_stretched_exponential(df[vcol],df[icol],efit,current_convention=cur_conv)
                elif advanced_model=='smoothed':
                    advanced_result=estimate_smoothed_plasma_potential_candidates(df[vcol],df[icol],window_length=int(sg_window),polynomial_order=int(sg_order),current_convention=cur_conv)
                else:
                    advanced_result=langmuir_fit_window_sensitivity(df[vcol],df[icol],efit,analysis_options={'current_convention':cur_conv})
                st.json(advanced_result)
            except Exception as exc: st.error(f'Advanced Langmuir analysis failed: {exc}')
    with st.expander('Separate electron-density and ion-flux estimates'):
        st.caption('These are separate model calculations, not automatic direct measurements from the canonical I–V fit.')
        estimate_kind=st.selectbox('Estimate',['Electron density from electron-saturation current','Multispecies ion density from ion-saturation current','Ion particle flux from supplied density'],key='probe_estimate_kind')
        if estimate_kind=='Electron density from electron-saturation current':
            ec1,ec2,ec3=st.columns(3)
            with ec1: electron_sat_current=st.number_input('Electron saturation current (A)',value=1e-3,format='%.6g',key='probe_e_sat_current')
            with ec2: electron_sat_te=st.number_input('Electron temperature (eV)',min_value=1e-6,value=5.0,key='probe_e_sat_te')
            with ec3: electron_sat_area=st.number_input('Effective collection area (m²)',min_value=1e-20,value=1e-4,format='%.6g',key='probe_e_sat_area')
            eu1,eu2,eu3=st.columns(3)
            with eu1: electron_sat_sigma=st.number_input('Current σ (A; 0 = not supplied)',min_value=0.0,value=0.0,format='%.6g',key='probe_e_sat_sigma')
            with eu2: electron_sat_te_rel=st.number_input('Te relative σ',min_value=0.0,value=0.0,format='%.6g',key='probe_e_sat_te_rel')
            with eu3: electron_sat_area_rel=st.number_input('Area relative σ',min_value=0.0,value=0.0,format='%.6g',key='probe_e_sat_area_rel')
            if st.button('Estimate electron density',key='probe_estimate_ne'):
                try: st.json(estimate_electron_density_from_saturation(electron_sat_current,electron_sat_te,electron_sat_area,current_convention=cur_conv,
                    current_standard_uncertainty_A=electron_sat_sigma if electron_sat_sigma>0 else None,
                    temperature_relative_standard_uncertainty=electron_sat_te_rel if electron_sat_te_rel>0 else None,
                    area_relative_standard_uncertainty=electron_sat_area_rel if electron_sat_area_rel>0 else None))
                except Exception as exc: st.error(str(exc))
        elif estimate_kind=='Multispecies ion density from ion-saturation current':
            mc1,mc2,mc3=st.columns(3)
            with mc1: multi_ion_current=st.number_input('Ion-saturation current (A)',value=-1e-3,format='%.6g',key='probe_multi_ion_current')
            with mc2: multi_te=st.number_input('Electron temperature (eV)',min_value=1e-6,value=5.0,key='probe_multi_te')
            with mc3: multi_area=st.number_input('Effective collection area (m²)',min_value=1e-20,value=1e-4,format='%.6g',key='probe_multi_area')
            multi_species_text=st.text_area(
                'Species JSON (fraction = number-density fraction; optional mass_relative_standard_uncertainty and sheath_coefficient_relative_standard_uncertainty)',
                value='[{"mass_kg": 3.34e-27, "fraction": 1.0, "charge_state": 1, "sheath_coefficient": 0.61}]',
                key='probe_multi_species_json')
            mu1,mu2,mu3=st.columns(3)
            with mu1: multi_current_sigma=st.number_input('Ion-current σ (A; 0 = not supplied)',min_value=0.0,value=0.0,format='%.6g',key='probe_multi_current_sigma')
            with mu2: multi_te_rel=st.number_input('Te relative σ (0 = not supplied)',min_value=0.0,value=0.0,format='%.6g',key='probe_multi_te_rel')
            with mu3: multi_area_rel=st.number_input('Area relative σ (0 = not supplied)',min_value=0.0,value=0.0,format='%.6g',key='probe_multi_area_rel')
            st.caption('Only explicitly supplied independent components are propagated. Species number-fraction covariance is excluded; fractions are treated as fixed. Add per-species mass/sheath-coefficient relative uncertainties in the JSON when available.')
            if st.button('Estimate multispecies ion density',key='probe_estimate_multi_ni'):
                try:
                    multi_species=json.loads(multi_species_text)
                    st.json(estimate_multispecies_ion_density(
                        multi_ion_current,multi_te,multi_area,multi_species,
                        current_convention=cur_conv,fraction_basis='number_density',
                        current_standard_uncertainty_A=multi_current_sigma if multi_current_sigma>0 else None,
                        temperature_relative_standard_uncertainty=multi_te_rel if multi_te_rel>0 else None,
                        area_relative_standard_uncertainty=multi_area_rel if multi_area_rel>0 else None))
                except Exception as exc: st.error(str(exc))
        else:
            fc1,fc2,fc3,fc4=st.columns(4)
            with fc1: flux_density=st.number_input('Ion density (m⁻³)',min_value=1e-6,value=1e18,format='%.6g',key='probe_flux_density')
            with fc2: flux_te=st.number_input('Electron temperature (eV)',min_value=1e-6,value=5.0,key='probe_flux_te')
            with fc3: flux_mass=st.number_input('Ion mass (kg)',min_value=1e-30,value=3.34e-27,format='%.6g',key='probe_flux_mass')
            with fc4: flux_factor=st.number_input('Sheath factor',min_value=1e-6,value=1.0,key='probe_flux_factor')
            flux_charge=st.number_input('Ion charge state Zᵢ',min_value=1,value=1,step=1,key='probe_flux_charge')
            fu1,fu2,fu3,fu4=st.columns(4)
            with fu1: flux_n_rel=st.number_input('Density relative σ',min_value=0.0,value=0.0,key='probe_flux_n_rel')
            with fu2: flux_te_rel=st.number_input('Te relative σ',min_value=0.0,value=0.0,key='probe_flux_te_rel')
            with fu3: flux_mass_rel=st.number_input('Mass relative σ',min_value=0.0,value=0.0,key='probe_flux_mass_rel')
            with fu4: flux_factor_rel=st.number_input('Sheath-factor relative σ',min_value=0.0,value=0.0,key='probe_flux_factor_rel')
            if st.button('Estimate ion particle flux',key='probe_estimate_flux'):
                try: st.json(estimate_ion_particle_flux(flux_density,flux_te,flux_mass,sheath_factor=flux_factor,charge_state=int(flux_charge),
                    density_relative_standard_uncertainty=flux_n_rel if flux_n_rel>0 else None,
                    temperature_relative_standard_uncertainty=flux_te_rel if flux_te_rel>0 else None,
                    mass_relative_standard_uncertainty=flux_mass_rel if flux_mass_rel>0 else None,
                    sheath_factor_relative_standard_uncertainty=flux_factor_rel if flux_factor_rel>0 else None))
                except Exception as exc: st.error(str(exc))
    with st.expander('Multi-channel independent I–V batch'):
        st.caption('Select multiple calibrated current columns sharing the selected voltage grid. Each channel is analyzed independently.')
        batch_current_cols=st.multiselect('Current columns (one per channel)',[c for c in numeric if c!=vcol],default=[icol],key='probe_batch_current_cols')
        if st.button('Analyze selected channels',key='probe_run_channel_batch'):
            try:
                if not batch_current_cols: raise ValueError('Select at least one current channel.')
                batch_currents=df[batch_current_cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float).T
                batch_result=analyze_langmuir_probe_array(df[vcol],batch_currents,channel_names=[str(x) for x in batch_current_cols],
                    current_convention=cur_conv,current_uncertainty_A=(sigma_current if sigma_current>0 else None),
                    electron_fit_range=efit,ion_fit_range=ifit)
                st.json(batch_result)
                json_download('Export multi-channel Langmuir JSON',batch_result,'tis_langmuir_multichannel.json')
            except Exception as exc: st.error(f'Multi-channel Langmuir analysis failed: {exc}')
    with st.expander('Time × channel independent sweep series'):
        st.caption('Long-form input schema: one row per sample, with time, channel, calibrated voltage and calibrated current columns. Every time/channel group must contain the same number of samples; no temporal trend or cross-channel inference is performed.')
        tcol=col_select('Time label column',numeric,'probe_series_time_col')
        channel_column_options=[c for c in df.columns if c not in {tcol,vcol,icol}]
        ccol=st.selectbox('Channel label column',channel_column_options or ['No channel label column available'],key='probe_series_channel_col')
        if not channel_column_options: st.info('Add/select a channel-label column to use time × channel batching.')
        svcol=col_select('Series voltage column',numeric,'probe_series_voltage_col',default=vcol)
        sicol=col_select('Series current column',numeric,'probe_series_current_col',default=icol)
        if st.button('Analyze time × channel series',key='probe_run_series',disabled=not channel_column_options):
            try:
                frame=df[[tcol,ccol,svcol,sicol]].copy()
                frame[tcol]=pd.to_numeric(frame[tcol],errors='coerce')
                frame[svcol]=pd.to_numeric(frame[svcol],errors='coerce')
                frame[sicol]=pd.to_numeric(frame[sicol],errors='coerce')
                if frame.isna().any().any(): raise ValueError('Time, channel, voltage and current values must be complete and numeric.')
                times=sorted(frame[tcol].unique().tolist())
                channel_labels=sorted(frame[ccol].astype(str).unique().tolist())
                voltage_rows=[]; current_rows=[]
                for time_value in times:
                    v_channels=[]; i_channels=[]
                    for channel_label in channel_labels:
                        group=frame[(frame[tcol]==time_value)&(frame[ccol].astype(str)==channel_label)].sort_values(svcol)
                        if len(group)<8: raise ValueError(f'Time {time_value}, channel {channel_label} has fewer than eight samples.')
                        v_channels.append(group[svcol].to_numpy(dtype=float))
                        i_channels.append(group[sicol].to_numpy(dtype=float))
                    if len({len(x) for x in v_channels})!=1: raise ValueError(f'All channels at time {time_value} must have the same sample count.')
                    voltage_rows.append(v_channels); current_rows.append(i_channels)
                if len({len(samples) for time_row in voltage_rows for samples in time_row})!=1:
                    raise ValueError('Every time/channel sweep must have the same sample count to form a rectangular time × channel × sample array.')
                series_v=np.asarray(voltage_rows,dtype=float); series_i=np.asarray(current_rows,dtype=float)
                series_result=analyze_langmuir_probe_series(series_v,series_i,time_labels=times,channel_names=channel_labels,
                    analysis_options={'current_convention':cur_conv,'current_uncertainty_A':(sigma_current if sigma_current>0 else None),
                                      'electron_fit_range':efit,'ion_fit_range':ifit})
                st.json(series_result)
                json_download('Export Langmuir time-series JSON',series_result,'tis_langmuir_series.json')
            except Exception as exc: st.error(f'Langmuir time-series analysis failed: {exc}')
    st.warning('All inputs are treated as already calibrated. The software does not calibrate data or perform empirical validation. Numerical fit screens and model estimates do not certify physical probe regimes; missing uncertainty inputs remain unquantified.')

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
        ece_mode=st.selectbox('ECE workflow',['Brightness temperature','Channel fluctuations / PSD','Resonance mapping','Temperature profile reconstruction','Receiver response convolution'],key='ece_workflow_mode')
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
                    st.dataframe(pd.DataFrame({'Frequency (GHz)':ece_freq_GHz,'Resonance field (T)':result['resonance_field_T'],'Brightness temperature (K)':result['brightness_temperature_K'],'Brightness temperature (eV)':result['brightness_temperature_eV']}),width="stretch",hide_index=True)
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
                    ece_welch_nperseg=st.number_input('Welch segment length (samples; 0 = automatic)',min_value=0,max_value=1000000,value=0,step=16,key='ece_welch_nperseg')
                    if st.button('Analyze ECE fluctuations / PSD',type='primary',key='ece_run_fluctuations'):
                        if len(ece_signal_cols)!=len(ece_freq_Hz):
                            raise ValueError('Select exactly one signal column per channel frequency.')
                        ece_frame=df[[ece_time_col]+ece_signal_cols].apply(pd.to_numeric,errors='coerce')
                        if ece_frame.isna().any().any():
                            raise ValueError('ECE time and selected channel columns must be numeric and complete. Missing/non-numeric samples are not dropped because doing so can corrupt the timebase and FFT spectrum; clean or explicitly resample the data first.')
                        ece_t=ece_frame[ece_time_col].to_numpy(dtype=float)
                        ece_y=ece_frame[ece_signal_cols].to_numpy(dtype=float)
                        ece_kwargs={'noise_std':ece_noise} if ece_noise>0 else {}
                        if ece_welch_nperseg>0:
                            ece_kwargs['welch_nperseg']=int(ece_welch_nperseg)
                        result=analyze_ece(ece_freq_Hz,harmonic=int(harm),signal=ece_y,time_s=ece_t,mode='channel_fluctuations',**ece_kwargs)
                        st.success('ECE fluctuation analysis completed.')
                        st.metric('Sampling frequency',f"{result['sampling_frequency_Hz']:.6g} Hz")
                        st.metric('Nyquist frequency',f"{result['nyquist_frequency_Hz']:.6g} Hz")
                        rms_frame=pd.DataFrame({'Frequency (GHz)':ece_freq_GHz,'Fluctuation RMS':result['fluctuation_rms']})
                        st.markdown('**Per-channel fluctuation RMS**')
                        st.dataframe(rms_frame,width="stretch",hide_index=True)
                        st.bar_chart(rms_frame.set_index('Frequency (GHz)')['Fluctuation RMS'])
                        psd_frame=pd.DataFrame(result['fluctuation_psd'],index=result['fluctuation_psd_frequency_Hz'],columns=[f'{v:g} GHz' for v in ece_freq_GHz])
                        psd_frame.index.name='Frequency (Hz)'
                        st.markdown('**Full-record one-sided periodogram**')
                        st.line_chart(psd_frame)
                        welch_frame=pd.DataFrame(result['welch_psd'],index=result['welch_psd_frequency_Hz'],columns=[f'{v:g} GHz' for v in ece_freq_GHz])
                        welch_frame.index.name='Frequency (Hz)'
                        st.markdown('**Welch-averaged PSD (Hann window; 50% overlap)**')
                        st.caption(f"Welch settings: {result['welch_psd_nperseg']} samples per segment; {result['welch_psd_overlap_samples']} samples overlap. The periodogram is retained for comparison.")
                        st.line_chart(welch_frame)
                        st.caption(result['aliasing_assessment'])
                        if result['stationarity_flag']:
                            st.warning('Split-record stationarity audit flagged material changes in mean/RMS. Welch confidence intervals assume more stationarity than this audit can establish; inspect time-resolved windows before interpreting peaks.')
                        else:
                            st.info('The split-record audit did not flag large mean/RMS changes; this is not proof of stationarity.')
                        if 'cross_channel_spectra' in result:
                            st.markdown('**Cross-channel coherence and phase (diagnostic statistics, not mode identification)**')
                            pair_choices=[f"{p['channel_i']}–{p['channel_j']}" for p in result['cross_channel_spectra']]
                            pair_index=st.selectbox('Channel pair (indices in selected-channel order)',range(len(pair_choices)),format_func=lambda idx:pair_choices[idx],key='ece_cross_pair')
                            pair=result['cross_channel_spectra'][pair_index]
                            cross_frame=pd.DataFrame({'Coherence':pair['coherence'],'Cross phase (rad)':pair['cross_phase_rad']},index=pair['coherence_frequency_Hz'])
                            cross_frame.index.name='Frequency (Hz)'
                            st.line_chart(cross_frame)
                            st.caption(result['cross_channel_spectra_note'])
                            st.caption(f"Estimated relative delay: {pair['estimated_relative_delay_s']} s. This phase-slope estimate is not hardware synchronization calibration.")
                        if result.get('spectral_peak_candidates'):
                            st.markdown('**Candidate spectral peaks (not automatic MHD-mode identification)**')
                            st.dataframe(pd.DataFrame(result['spectral_peak_candidates']),width="stretch",hide_index=True)
                            st.caption(result['spectral_peak_note'])
                        if 'above_noise_rms' in result:
                            st.dataframe(pd.DataFrame({'Frequency (GHz)':ece_freq_GHz,'RMS above noise':result['above_noise_rms']}),width="stretch",hide_index=True)
                        st.json({k:v for k,v in result.items() if k not in {'fluctuation_psd','fluctuation_psd_frequency_Hz','welch_psd','welch_psd_frequency_Hz','time_s'}})
                        json_download('Export ECE fluctuation / PSD result',result,'tis_ece_fluctuations.json')
            elif ece_mode=='Resonance mapping':
                ece_r0=st.number_input('Reference major radius R₀ (m)',min_value=1e-6,value=1.0,key='ece_r0')
                ece_b0=st.number_input('Toroidal field at R₀ (T)',min_value=1e-6,value=2.0,key='ece_b0')
                ece_map_rel=st.checkbox('Use monoenergetic relativistic/Doppler correction (approximate)',value=False,key='ece_map_rel')
                ece_map_energy=st.number_input('Representative electron kinetic energy (eV)',min_value=0.0,value=1000.0,key='ece_map_energy') if ece_map_rel else None
                ece_map_beta=st.number_input('Signed line-of-sight beta',min_value=-0.99,max_value=0.99,value=0.0,key='ece_map_beta') if ece_map_rel else 0.0
                if st.button('Map ECE resonance radii',type='primary',key='ece_run_mapping'):
                    result=analyze_ece(ece_freq_Hz,harmonic=int(harm),major_radius_m=ece_r0,B_axis_T=ece_b0,mode='resonance_mapping',electron_energy_eV=ece_map_energy,parallel_beta=ece_map_beta)
                    st.success('ECE resonance mapping completed using the declared circular toroidal-field approximation.')
                    st.dataframe(pd.DataFrame({'Frequency (GHz)':ece_freq_GHz,'Resonance field (T)':result['resonance_field_T'],'Approximate major radius (m)':result['resonance_radius_m']}),width="stretch",hide_index=True)
                    st.line_chart(pd.DataFrame({'Approximate major radius (m)':result['resonance_radius_m']},index=ece_freq_GHz))
                    st.warning('This is a 1-D circular-field estimate, not EFIT/ray tracing. A monoenergetic relativistic/Doppler correction is only a sensitivity estimate; distribution effects, line-of-sight geometry and instrument response remain unmodeled.')
                    st.json(result)
                    json_download('Export ECE resonance mapping',result,'tis_ece_resonance_mapping.json')
            elif ece_mode=='Receiver response convolution':
                st.markdown('**Convolve calibrated spectral radiance with a supplied receiver response**')
                st.caption('Supply an already-calibrated spectrum and the measured/validated response kernel. TIS does not infer the response function or perform detector calibration.')
                ece_spectrum_frequency_text=st.text_area('Spectral frequency grid (Hz; comma-separated, increasing)',key='ece_response_spectrum_frequency')
                ece_spectrum_radiance_text=st.text_area('Calibrated spectral radiance values (same length; W m⁻² sr⁻¹ Hz⁻¹)',key='ece_response_spectrum_radiance')
                ece_response_frequency_text=st.text_area('Response frequency grid (Hz; comma-separated, increasing)',key='ece_response_frequency')
                ece_response_weights_text=st.text_area('Response weights: one comma-separated row per channel, rows separated by semicolons',key='ece_response_weights')
                if st.button('Convolve spectrum with receiver response',type='primary',key='ece_response_run'):
                    try:
                        spec_f=np.asarray([float(v.strip()) for v in ece_spectrum_frequency_text.split(',') if v.strip()],dtype=float)
                        spec_rad=np.asarray([float(v.strip()) for v in ece_spectrum_radiance_text.split(',') if v.strip()],dtype=float)
                        response_f=np.asarray([float(v.strip()) for v in ece_response_frequency_text.split(',') if v.strip()],dtype=float)
                        response_rows=[[float(v.strip()) for v in row.split(',') if v.strip()] for row in ece_response_weights_text.split(';') if row.strip()]
                        response_w=np.asarray(response_rows,dtype=float)
                        result=convolve_ece_receiver_response(spec_f,spec_rad,response_f,response_w)
                        result['channel_brightness_temperature_K']=brightness_temperature_K(result['channel_spectral_radiance_W_m2_sr_Hz'],result['effective_center_frequency_Hz'])
                        st.success('ECE receiver-response convolution completed using the supplied kernel.')
                        st.caption('Brightness temperature evaluated at the response-weighted center frequency is approximate for broad channels.')
                        st.dataframe(pd.DataFrame({'Channel':np.arange(response_w.shape[0]),'Effective spectral radiance (W m⁻² sr⁻¹ Hz⁻¹)':result['channel_spectral_radiance_W_m2_sr_Hz'],'Brightness temperature (K)':result['channel_brightness_temperature_K']}),width="stretch",hide_index=True)
                        st.json(result)
                        json_download('Export ECE receiver-response convolution',result,'tis_ece_receiver_response.json')
                    except Exception as e:
                        st.error(f'ECE receiver-response convolution failed: {e}')
            else:
                st.markdown('**Radial electron-temperature profile from multi-channel ECE data**')
                st.caption('Inputs are already calibrated brightness temperatures in eV. TIS performs no detector calibration or signal-to-temperature conversion.')
                if df is None:
                    st.info('Upload a table with a time column and one already-calibrated brightness-temperature (eV) column per ECE channel.')
                else:
                    ece_time_col=col_select('Profile time column (s)',list(df.columns),'ece_profile_time_col')
                    ece_signal_cols=st.multiselect('Calibrated brightness-temperature columns (eV; same order as frequencies)',[col for col in numeric if col!=ece_time_col],key='ece_profile_signal_cols')
                    ece_tau_text=st.text_input('Optical depth per channel (comma-separated; blank if assuming optically thick)',key='ece_profile_tau')
                    ece_thick=st.checkbox('I confirm the channels are optically thick; use Te ≈ Tb',value=False,key='ece_profile_thick')
                    ece_bg=st.number_input('Background brightness temperature (eV)',min_value=0.0,value=0.0,key='ece_profile_background')
                    ece_bandwidth_profile_text=st.text_input('Optional channel bandwidths (Hz; one value or one per channel)',key='ece_profile_bandwidth')
                    ece_tb_sigma_text=st.text_input('Optional brightness-temperature 1-sigma uncertainty (eV; one value or one per channel)',key='ece_profile_tb_sigma')
                    ece_freq_sigma_text=st.text_input('Optional center-frequency 1-sigma uncertainty (Hz; one value or one per channel)',key='ece_profile_freq_sigma')
                    ece_tau_sigma_text=st.text_input('Optional optical-depth 1-sigma uncertainty (one value or one per channel)',key='ece_profile_tau_sigma')
                    ece_bg_sigma=st.number_input('Background-temperature 1-sigma uncertainty (eV)',min_value=0.0,value=0.0,key='ece_profile_bg_sigma')
                    ece_r0_sigma=st.number_input('Reference-radius 1-sigma uncertainty (m)',min_value=0.0,value=0.0,key='ece_profile_r0_sigma')
                    ece_b0_sigma=st.number_input('Reference-field 1-sigma uncertainty (T)',min_value=0.0,value=0.0,key='ece_profile_b0_sigma')
                    ece_covariance_text=st.text_area('Optional covariance matrix of time-window-mean Tb channels (eV²; rows separated by semicolons)',key='ece_profile_covariance')
                    ece_time_resolved=st.checkbox('Also return time-resolved Te(R,t) and uncertainty estimates',value=False,key='ece_profile_time_resolved')
                    ece_time_values=pd.to_numeric(df[ece_time_col],errors='coerce').dropna()
                    if ece_time_values.empty:
                        raise ValueError('The selected profile time column has no numeric samples.')
                    ece_t_min=float(ece_time_values.min())
                    ece_t_max=float(ece_time_values.max())
                    st.caption('The profile is time-averaged only over this explicit acquisition window; it is not a time-resolved Te profile.')
                    ece_window_start=st.number_input('Profile window start (s)',min_value=ece_t_min,max_value=ece_t_max,value=ece_t_min,key='ece_profile_window_start')
                    ece_window_end=st.number_input('Profile window end (s)',min_value=ece_t_min,max_value=ece_t_max,value=ece_t_max,key='ece_profile_window_end')
                    ece_r0=st.number_input('Reference major radius R₀ (m)',min_value=1e-6,value=1.0,key='ece_profile_r0')
                    ece_b0=st.number_input('Toroidal field at R₀ (T)',min_value=1e-6,value=2.0,key='ece_profile_b0')
                    ece_profile_method=st.selectbox('Profile interpolation method',['Shape-preserving PCHIP','Linear'],key='ece_profile_method')
                    ece_model_floor=st.number_input('Optional fractional model-discrepancy floor (0 = not quantified)',min_value=0.0,max_value=1.0,value=0.0,step=0.01,format='%.3f',key='ece_model_discrepancy')
                    ece_relativistic=st.checkbox('Apply monoenergetic relativistic/Doppler resonance correction (approximate)',value=False,key='ece_relativistic')
                    ece_energy_eV=0.0
                    ece_beta_parallel=0.0
                    if ece_relativistic:
                        ece_energy_eV=st.number_input('Representative electron kinetic energy (eV)',min_value=0.0,value=1000.0,format='%.6g',key='ece_relativistic_energy')
                        ece_beta_parallel=st.number_input('Signed line-of-sight beta (v_parallel/c; -1 < beta < 1)',min_value=-0.99,max_value=0.99,value=0.0,step=0.01,format='%.4f',key='ece_relativistic_beta')
                        st.warning('Monoenergetic correction only; it is not a distribution-averaged emissivity, ray-tracing or relativistic radiative-transfer model.')
                    ece_eq_radius_text=st.text_input('Optional equilibrium R grid (m; comma-separated; blank uses circular-field approximation)',key='ece_profile_eq_radius')
                    ece_eq_bphi_text=st.text_input('Matching equilibrium toroidal Bφ grid (T; strictly monotonic; blank uses circular-field approximation)',key='ece_profile_eq_bphi')
                    ece_explicit_radius_text=st.text_input('Optional externally ray-traced / equilibrium-derived channel resonance radii (m; one per channel; overrides the 1-D model)',key='ece_profile_explicit_radii')
                    ece_radius_unc_text=st.text_input('Optional 1-sigma channel-radius uncertainty (m; one value or one per channel)',key='ece_profile_radius_uncertainty')
                    ece_plasma_bounds_text=st.text_input('Optional physical plasma major-radius bounds (R_min,R_max in m; validates mapped channels)',key='ece_profile_plasma_bounds')
                    if st.button('Reconstruct ECE electron-temperature profile',type='primary',key='ece_run_profile'):
                        if len(ece_signal_cols)!=len(ece_freq_Hz):
                            raise ValueError('Select exactly one calibrated brightness-temperature column per channel frequency.')
                        ece_tau=None if not ece_tau_text.strip() else np.asarray([float(v.strip()) for v in ece_tau_text.split(',') if v.strip()],dtype=float)
                        if ece_tau is None and not ece_thick:
                            raise ValueError('Supply optical depth per channel or explicitly confirm the optically thick assumption.')
                        ece_frame=df[[ece_time_col]+ece_signal_cols].apply(pd.to_numeric,errors='coerce')
                        if ece_frame.isna().any().any():
                            raise ValueError('Profile time and channel signals must be numeric and complete; missing samples are not silently dropped.')
                        eq_r=None if not ece_eq_radius_text.strip() else np.asarray([float(v.strip()) for v in ece_eq_radius_text.split(',') if v.strip()],dtype=float)
                        eq_b=None if not ece_eq_bphi_text.strip() else np.asarray([float(v.strip()) for v in ece_eq_bphi_text.split(',') if v.strip()],dtype=float)
                        if (eq_r is None)!=(eq_b is None):
                            raise ValueError('Supply both equilibrium R and Bφ arrays, or leave both blank.')
                        explicit_r=None if not ece_explicit_radius_text.strip() else np.asarray([float(v.strip()) for v in ece_explicit_radius_text.split(',') if v.strip()],dtype=float)
                        radius_unc=None if not ece_radius_unc_text.strip() else np.asarray([float(v.strip()) for v in ece_radius_unc_text.split(',') if v.strip()],dtype=float)
                        plasma_bounds=None if not ece_plasma_bounds_text.strip() else np.asarray([float(v.strip()) for v in ece_plasma_bounds_text.split(',') if v.strip()],dtype=float)
                        if plasma_bounds is not None and plasma_bounds.shape!=(2,):
                            raise ValueError('Enter exactly R_min,R_max for physical plasma major-radius bounds.')
                        if explicit_r is not None and eq_r is not None:
                            raise ValueError('Use either explicit channel resonance radii or an equilibrium R/Bφ table, not both.')
                        bandwidth=None if not ece_bandwidth_profile_text.strip() else np.asarray([float(v.strip()) for v in ece_bandwidth_profile_text.split(',') if v.strip()],dtype=float)
                        tb_sigma=None if not ece_tb_sigma_text.strip() else np.asarray([float(v.strip()) for v in ece_tb_sigma_text.split(',') if v.strip()],dtype=float)
                        freq_sigma=None if not ece_freq_sigma_text.strip() else np.asarray([float(v.strip()) for v in ece_freq_sigma_text.split(',') if v.strip()],dtype=float)
                        tau_sigma=None if not ece_tau_sigma_text.strip() else np.asarray([float(v.strip()) for v in ece_tau_sigma_text.split(',') if v.strip()],dtype=float)
                        covariance=None if not ece_covariance_text.strip() else np.asarray([[float(v.strip()) for v in row.split(',') if v.strip()] for row in ece_covariance_text.split(';') if row.strip()],dtype=float)
                        result=reconstruct_ece_temperature_profile(
                            ece_freq_Hz,ece_frame[ece_signal_cols].to_numpy(dtype=float),
                            ece_frame[ece_time_col].to_numpy(dtype=float),
                            major_radius_m=ece_r0,B_axis_T=ece_b0,harmonic=int(harm),
                            optical_depth=ece_tau,background_temperature_eV=ece_bg,
                            assume_optically_thick=ece_thick,
                            time_window_s=(ece_window_start,ece_window_end),
                            profile_method='pchip' if ece_profile_method=='Shape-preserving PCHIP' else 'linear',
                            equilibrium_radius_m=eq_r,equilibrium_toroidal_field_T=eq_b,
                            channel_resonance_radius_m=explicit_r,
                            radius_mapping_uncertainty_m=radius_unc,
                            plasma_radius_bounds_m=plasma_bounds,
                            channel_bandwidth_Hz=bandwidth,
                            brightness_temperature_uncertainty_eV=tb_sigma,
                            frequency_uncertainty_Hz=freq_sigma,
                            optical_depth_uncertainty=tau_sigma,
                            background_temperature_uncertainty_eV=float(ece_bg_sigma),
                            major_radius_uncertainty_m=float(ece_r0_sigma),
                            B_axis_uncertainty_T=float(ece_b0_sigma),
                            channel_covariance_eV2=covariance,
                            include_time_resolved_profile=ece_time_resolved,
                            electron_energy_eV=(ece_energy_eV if ece_relativistic else None),
                            parallel_beta=ece_beta_parallel,
                            model_discrepancy_fraction=float(ece_model_floor),
                        )
                        st.success(f'ECE time-window-averaged electron-temperature profile completed ({result["selected_sample_count"]} samples; {ece_window_start:g}–{ece_window_end:g} s) within the declared model.')
                        st.dataframe(pd.DataFrame({'Frequency (GHz)':ece_freq_GHz,'Mean calibrated brightness temperature (eV)':result['mean_calibrated_brightness_temperature_eV'],'Inferred Te (eV)':result['electron_temperature_channel_eV'],'Approximate resonance radius (m)':result['resonance_radius_m']}),width="stretch",hide_index=True)
                        st.line_chart(pd.DataFrame({'Inferred Te profile (eV)':result['electron_temperature_profile_eV']},index=result['profile_radius_m']))
                        st.line_chart(pd.DataFrame({'dTe/dR (eV/m)':result['profile_gradient_eV_per_m']},index=result['profile_radius_m']))
                        if result['electron_temperature_profile_uncertainty_eV'] is not None:
                            st.line_chart(pd.DataFrame({'Te (eV)':result['electron_temperature_profile_eV'],'Approximate 1-sigma uncertainty (eV)':result['electron_temperature_profile_uncertainty_eV']},index=result['profile_radius_m']))
                        if 'electron_temperature_profile_time_eV' in result:
                            st.markdown('**Time-resolved radial Te**')
                            st.line_chart(pd.DataFrame(result['electron_temperature_profile_time_eV'].T,index=result['profile_radius_m']))
                            if 'electron_temperature_profile_time_uncertainty_eV' in result:
                                st.caption('Time-resolved uncertainty includes supplied measurement/model-input errors and radial-position sensitivity; it is still first-order.')
                        st.caption(f"Measured channels: {result['profile_spatial_quality']['measured_channel_count']} · Output grid points (not independent measurements): {result['profile_spatial_quality']['output_grid_point_count']} · Max PCHIP-vs-linear difference: {result['profile_interpolation_method_sensitivity']}")
                        st.warning('Brightness temperature is not automatically electron temperature. The inversion depends on optical-depth/thermalization assumptions. Uncertainty bands omit unquantified physics unless a model-discrepancy floor is supplied.')
                        st.warning('Radius mapping is one-dimensional. A supplied Bφ(R) table is not full EFIT flux-coordinate mapping or line-of-sight ray tracing; verify device bounds and viewing geometry independently.')
                        st.json(result)
                        json_download('Export ECE electron-temperature profile',result,'tis_ece_temperature_profile.json')
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
        st.caption('Enter a neutron source rate already calibrated and corrected for detector efficiency and geometry. TIS applies no calibration factors.')
        source_rate=st.number_input('Calibrated neutron source rate (s⁻¹)',min_value=0.0,value=1000.0,key='calibrated_neutron_source_rate')
        if st.button('Report calibrated neutron source rate'):
            result={'diagnostic':'Neutron source rate','neutron_source_rate_s-1':float(source_rate),'input_status':'user-supplied calibrated value','calibration_performed':False}
            st.json(result); json_download('Export calibrated neutron source rate',result,'tis_neutron_rate.json')

elif page=='Equilibrium & Reconstruction':
    st.title('Equilibrium & Magnetic Reconstruction Tools')
    st.caption('Run a Grad–Shafranov reconstruction from a public DIII-D EFIT flux map or solve from your own boundary and pressure/current profiles.')
    st.markdown('### Public-data equilibrium reconstruction')
    st.caption('Fetch a real EFIT poloidal-flux map from the open Fusion Equilibrium Challenge dataset, infer a flux-dependent Grad–Shafranov source profile, solve with the reference perimeter fixed, and compare the reconstructed interior against the reference.')
    public_eq_col1, public_eq_col2 = st.columns(2)
    with public_eq_col1:
        public_eq_sample = st.number_input('Public DIII-D sample index', min_value=0, value=0, step=1, key='public_eq_sample')
    with public_eq_col2:
        public_eq_time = st.number_input('EFIT time (s)', min_value=0.0, value=0.2, format='%.6f', key='public_eq_time')
    public_eq_f_edge = st.number_input(
        'Edge F = R Bφ (T·m; enter 0 to omit q)',
        min_value=-1000.0, max_value=1000.0, value=0.0, format='%.6g',
        help='Use the independently known EFIT edge value of F=R·B_toroidal in T·m. Do not enter a guessed value; q(ψ) is omitted when this is zero.',
        key='public_eq_f_edge',
    )
    if st.button('Fetch public EFIT map and reconstruct', type='primary', key='public_eq_reconstruct'):
        try:
            with st.spinner('Fetching public DIII-D EFIT map and solving Grad–Shafranov equation...'):
                from tokamak_analysis.equilibrium_reconstruction import (
                    reconstruct_equilibrium_from_reference_flux,
                    reconstruct_equilibrium_profiles_from_flux,
                    resample_flux_map_to_uniform_grid,
                )
                public_eq_R, public_eq_Z, public_eq_reference, public_eq_manifest = fetch_public_efit_flux_map(
                    int(public_eq_sample), float(public_eq_time)
                )
                was_nonuniform = (
                    not np.allclose(np.diff(public_eq_R), np.diff(public_eq_R)[0])
                    or not np.allclose(np.diff(public_eq_Z), np.diff(public_eq_Z)[0])
                )
                public_eq_R, public_eq_Z, public_eq_reference = resample_flux_map_to_uniform_grid(
                    public_eq_R, public_eq_Z, public_eq_reference
                )
                public_eq_manifest = {
                    **public_eq_manifest,
                    'solver_grid_resampled_uniform': bool(was_nonuniform),
                    'solver_grid_interpolation': 'separable linear interpolation in R then Z',
                }
                public_eq_profiles = reconstruct_equilibrium_profiles_from_flux(
                    public_eq_R, public_eq_Z, public_eq_reference,
                    n_profile_points=32,
                    toroidal_field_boundary_Tm=(float(public_eq_f_edge) if float(public_eq_f_edge) != 0.0 else None),
                )
                public_eq_result = reconstruct_equilibrium_from_reference_flux(
                    public_eq_R, public_eq_Z, public_eq_reference,
                    n_profile_points=32, max_iterations=5000,
                    tolerance=1e-6, relaxation=1.1,
                    toroidal_field_boundary_Tm=(float(public_eq_f_edge) if float(public_eq_f_edge) != 0.0 else None),
                    psi_profile=public_eq_profiles["psi_normalized_axis_to_edge"],
                    pressure_gradient_Pa_per_Wb=public_eq_profiles["pressure_gradient_Pa_per_Wb"],
                    ffprime_T2=public_eq_profiles["ffprime_T2"],
                )
                st.session_state['public_eq_result'] = public_eq_result
                st.session_state['public_eq_profiles'] = public_eq_profiles
                st.session_state['public_eq_manifest'] = public_eq_manifest
                st.session_state['public_eq_R'] = public_eq_R
                st.session_state['public_eq_Z'] = public_eq_Z
        except Exception as exc:
            st.error(f'Public-data equilibrium reconstruction failed: {exc}')
            st.info('The public Hugging Face dataset must be reachable and the optional data dependencies must be installed.')
    if 'public_eq_result' in st.session_state:
        public_eq_result = st.session_state['public_eq_result']
        public_eq_manifest = st.session_state['public_eq_manifest']
        public_eq_R = st.session_state['public_eq_R']
        public_eq_Z = st.session_state['public_eq_Z']
        public_eq_metrics = public_eq_result['reference_comparison']
        public_eq_profiles = st.session_state['public_eq_profiles']
        st.markdown('**Public dataset provenance**')
        st.json(public_eq_manifest)
        public_eq_m1, public_eq_m2, public_eq_m3 = st.columns(3)
        public_eq_m1.metric('Interior flux NRMSE', f"{public_eq_metrics['interior_flux_nrmse_by_reference_range']:.4g}")
        public_eq_m2.metric('Interior flux RMSE', f"{public_eq_metrics['interior_flux_rmse_Wb_rad']:.4g} Wb/rad")
        public_eq_m3.metric('Solver iterations', str(public_eq_result['iterations']))
        st.metric('Grad–Shafranov solve', 'Converged' if public_eq_result['converged'] else 'Not converged')
        public_eq_fig = Figure(figsize=(12, 4.5))
        public_eq_axes = public_eq_fig.subplots(1, 3)
        public_eq_fields = [
            (public_eq_result['reference_psi_Wb_rad'], 'Public EFIT reference flux'),
            (public_eq_result['psi_Wb_rad'], 'Reconstructed flux'),
            (public_eq_result['psi_Wb_rad'] - public_eq_result['reference_psi_Wb_rad'], 'Reconstruction − reference'),
        ]
        for public_eq_ax, (public_eq_field, public_eq_title) in zip(public_eq_axes, public_eq_fields):
            public_eq_mesh = public_eq_ax.pcolormesh(public_eq_R, public_eq_Z, public_eq_field, shading='auto')
            public_eq_ax.set_xlabel('R (m)')
            public_eq_ax.set_ylabel('Z (m)')
            public_eq_ax.set_title(public_eq_title)
            public_eq_fig.colorbar(public_eq_mesh, ax=public_eq_ax)
        public_eq_fig.tight_layout()
        st.pyplot(public_eq_fig)
        st.json({key: value for key, value in public_eq_result.items()
                 if key not in {'psi_Wb_rad', 'psi_normalized', 'B_R_T', 'B_Z_T',
                                'B_poloidal_T', 'reference_psi_Wb_rad'}})
        public_eq_table = pd.DataFrame({
            'Z_m': np.repeat(public_eq_Z, public_eq_R.size),
            'R_m': np.tile(public_eq_R, public_eq_Z.size),
            'reference_psi_Wb_rad': public_eq_result['reference_psi_Wb_rad'].ravel(),
            'reconstructed_psi_Wb_rad': public_eq_result['psi_Wb_rad'].ravel(),
            'psi_difference_Wb_rad': (public_eq_result['psi_Wb_rad'] - public_eq_result['reference_psi_Wb_rad']).ravel(),
            'B_R_T': public_eq_result['B_R_T'].ravel(),
            'B_Z_T': public_eq_result['B_Z_T'].ravel(),
            'B_poloidal_T': public_eq_result['B_poloidal_T'].ravel(),
        })
        download_df('Download public-data reconstruction and comparison', public_eq_table,
                    'tis_public_efit_reconstruction.csv')
        st.markdown('### Reconstructed equilibrium profiles')
        st.caption(
            'Pressure-gradient and F·F′ profiles are estimated jointly from the Grad–Shafranov operator and its R² dependence on each flux surface. '
            'This is an inverse-source estimate, not a unique pressure/current reconstruction; the dataset provides ψ but withholds EFIT profile labels.'
        )
        prof1, prof2, prof3 = st.columns(3)
        prof1.metric('Magnetic-axis R', f"{public_eq_profiles['axis_R_m']:.4g} m")
        prof2.metric('Magnetic-axis Z', f"{public_eq_profiles['axis_Z_m']:.4g} m")
        prof3.metric('Joint p′ / FF′ fit', 'Identifiable' if public_eq_profiles['joint_pressure_ffprime_identifiable'] else 'Underdetermined')
        profile_table_data = {
            'psi_normalized_axis_to_edge': public_eq_profiles['psi_normalized_axis_to_edge'],
            'pressure_gradient_Pa_per_Wb': public_eq_profiles['pressure_gradient_Pa_per_Wb'],
            'ffprime_T2': public_eq_profiles['ffprime_T2'],
            'pressure_Pa_relative_to_edge': public_eq_profiles['pressure_Pa_relative_to_edge'],
            'samples_in_flux_bin': public_eq_profiles['profile_bin_counts'],
            'source_fit_rank': public_eq_profiles['profile_bin_design_rank'],
        }
        if 'F_Tm' in public_eq_profiles:
            profile_table_data['F_Tm'] = public_eq_profiles['F_Tm']
            profile_table_data['q_profile'] = public_eq_profiles['q_profile']
            if public_eq_profiles['q95'] is not None:
                st.metric('Estimated q95 (requires supplied edge F)', f"{public_eq_profiles['q95']:.5g}")
            else:
                st.warning('q95 could not be estimated because the flux map did not provide enough closed interior flux contours.')
        else:
            st.info('q(ψ) is not calculated: enter an independently known edge F = R·Bφ value to constrain the toroidal field function.')
        profile_table = pd.DataFrame(profile_table_data)
        st.dataframe(profile_table, width="stretch")
        profile_fig = Figure(figsize=(9, 5))
        profile_ax = profile_fig.subplots()
        profile_ax.plot(profile_table_data['psi_normalized_axis_to_edge'],
                        profile_table_data['pressure_gradient_Pa_per_Wb'], label="p′(ψ)")
        profile_ax.plot(profile_table_data['psi_normalized_axis_to_edge'],
                        profile_table_data['ffprime_T2'], label="F F′(ψ)")
        profile_ax.set_xlabel('Normalized poloidal flux (axis 0 → edge 1)')
        profile_ax.set_ylabel('Inferred Grad–Shafranov source profile')
        profile_ax.legend()
        profile_fig.tight_layout()
        st.pyplot(profile_fig)
        download_df('Download reconstructed equilibrium profiles', profile_table,
                    'tis_equilibrium_profiles.csv')
        st.markdown('**Derived toroidal-current-density map**')
        st.caption('jφ = R p′(ψ) + F F′(ψ)/(μ₀R); values outside the inferred axis-to-edge flux interval are masked.')
        public_jtor_table = pd.DataFrame({
            'Z_m': np.repeat(public_eq_Z, public_eq_R.size),
            'R_m': np.tile(public_eq_R, public_eq_Z.size),
            'j_tor_A_m2': public_eq_profiles['j_tor_A_m2'].ravel(),
        })
        public_jtor_fig = Figure(figsize=(8, 4.5))
        public_jtor_ax = public_jtor_fig.subplots()
        public_jtor_mesh = public_jtor_ax.pcolormesh(
            public_eq_R, public_eq_Z, public_eq_profiles['j_tor_A_m2'],
            shading='auto',
        )
        public_jtor_ax.set_xlabel('R (m)')
        public_jtor_ax.set_ylabel('Z (m)')
        public_jtor_ax.set_title('Inferred toroidal current density')
        public_jtor_fig.colorbar(public_jtor_mesh, ax=public_jtor_ax, label='jφ (A/m²)')
        public_jtor_fig.tight_layout()
        st.pyplot(public_jtor_fig)
        download_df('Download toroidal-current-density grid', public_jtor_table,
                    'tis_equilibrium_toroidal_current_density.csv')
        st.json({key: value for key, value in public_eq_profiles.items()
                 if key not in {'operator_Wb_rad_m-2', 'psi_normalized_grid', 'j_tor_A_m2'}})
        st.warning(public_eq_result['warning'])
    st.divider()
    st.markdown('### Grad–Shafranov solver (prescribed boundary and profiles)')
    st.caption('Upload R and Z coordinate vectors, a boundary-flux grid shaped (len(Z), len(R)), and three profile CSV vectors: normalized flux spanning the full axis-to-edge interval 0→1, p\'(ψ), and F F\'(ψ). Partial normalized-flux profiles are rejected to prevent silent endpoint clamping.')
    with st.form('grad_shafranov_form'):
        eq_R_solver = st.file_uploader('Solver R coordinates (m; CSV vector)', type=['csv'], key='gs_R')
        eq_Z_solver = st.file_uploader('Solver Z coordinates (m; CSV vector)', type=['csv'], key='gs_Z')
        eq_boundary_solver = st.file_uploader('Boundary poloidal flux grid (Wb/rad; CSV matrix)', type=['csv'], key='gs_boundary')
        eq_profile_solver = st.file_uploader('Normalized-flux profile coordinates (CSV vector)', type=['csv'], key='gs_profile')
        eq_pprime_solver = st.file_uploader('Pressure derivative p\'(ψ) (Pa/Wb; CSV vector)', type=['csv'], key='gs_pprime')
        eq_ffprime_solver = st.file_uploader('F F\'(ψ) (T²; CSV vector)', type=['csv'], key='gs_ffprime')
        gs_tolerance = st.number_input('Relative iteration tolerance', min_value=1e-12, max_value=1e-2, value=1e-8, format='%.1e', key='gs_tolerance')
        gs_max_iterations = st.number_input('Maximum iterations', min_value=1, max_value=100000, value=10000, step=100, key='gs_max_iterations')
        gs_relaxation = st.number_input('Relaxation factor (0, 2)', min_value=0.01, max_value=1.99, value=1.2, step=0.05, key='gs_relaxation')
        gs_f_edge = st.number_input(
            'Edge F = R Bφ (T·m; 0 omits F and q)', min_value=-1000.0, max_value=1000.0,
            value=0.0, format='%.6g', key='gs_f_edge',
            help='Use an independently known edge value. Do not guess it.',
        )
        gs_run = st.form_submit_button('Solve Grad–Shafranov equilibrium', type='primary')
    if gs_run:
        try:
            from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium
            required_uploads = [eq_R_solver, eq_Z_solver, eq_boundary_solver, eq_profile_solver, eq_pprime_solver, eq_ffprime_solver]
            if any(upload is None for upload in required_uploads):
                raise ValueError('Upload all six CSV inputs before running the solver.')
            gs_R = pd.read_csv(eq_R_solver, header=None).to_numpy(dtype=float).ravel()
            gs_Z = pd.read_csv(eq_Z_solver, header=None).to_numpy(dtype=float).ravel()
            gs_boundary = pd.read_csv(eq_boundary_solver, header=None).to_numpy(dtype=float)
            gs_profile = pd.read_csv(eq_profile_solver, header=None).to_numpy(dtype=float).ravel()
            gs_pprime = pd.read_csv(eq_pprime_solver, header=None).to_numpy(dtype=float).ravel()
            gs_ffprime = pd.read_csv(eq_ffprime_solver, header=None).to_numpy(dtype=float).ravel()
            gs_result = reconstruct_equilibrium(
                gs_R, gs_Z, gs_boundary, gs_profile, gs_pprime, gs_ffprime,
                max_iterations=int(gs_max_iterations), tolerance=float(gs_tolerance),
                relaxation=float(gs_relaxation),
            )
            from tokamak_analysis.equilibrium_reconstruction import derive_equilibrium_profiles_from_sources
            gs_profiles = derive_equilibrium_profiles_from_sources(
                gs_R, gs_Z, gs_result['psi_Wb_rad'], gs_profile, gs_pprime, gs_ffprime,
                toroidal_field_boundary_Tm=(float(gs_f_edge) if float(gs_f_edge) != 0.0 else None),
            )
            st.metric('Solver convergence', 'Converged' if gs_result['converged'] else 'Not converged')
            st.json({key: value for key, value in gs_result.items() if key not in {'psi_Wb_rad', 'psi_normalized', 'B_R_T', 'B_Z_T', 'B_poloidal_T'}})
            gs_table = pd.DataFrame({'Z_m': np.repeat(gs_Z, gs_R.size), 'R_m': np.tile(gs_R, gs_Z.size),
                                     'psi_Wb_rad': gs_result['psi_Wb_rad'].ravel(),
                                     'psi_normalized': gs_result['psi_normalized'].ravel(),
                                     'B_R_T': gs_result['B_R_T'].ravel(),
                                     'B_Z_T': gs_result['B_Z_T'].ravel(),
                                     'B_poloidal_T': gs_result['B_poloidal_T'].ravel()})
            st.dataframe(gs_table.head(30), width="stretch")
            # Render the actual R-Z flux and poloidal-field structure, not only a table.
            gs_fig = Figure(figsize=(8, 5))
            gs_ax = gs_fig.subplots()
            gs_mesh = gs_ax.pcolormesh(gs_R, gs_Z, gs_result['psi_normalized'], shading='auto')
            gs_ax.set_xlabel('Major radius R (m)')
            gs_ax.set_ylabel('Vertical coordinate Z (m)')
            gs_ax.set_title('Normalized poloidal flux')
            gs_fig.colorbar(gs_mesh, ax=gs_ax, label='Normalized flux')
            gs_fig.tight_layout()
            st.pyplot(gs_fig)
            gs_field_fig = Figure(figsize=(8, 5))
            gs_field_ax = gs_field_fig.subplots()
            gs_field_mesh = gs_field_ax.pcolormesh(gs_R, gs_Z, gs_result['B_poloidal_T'], shading='auto')
            gs_field_ax.set_xlabel('Major radius R (m)')
            gs_field_ax.set_ylabel('Vertical coordinate Z (m)')
            gs_field_ax.set_title('Poloidal magnetic-field magnitude')
            gs_field_fig.colorbar(gs_field_mesh, ax=gs_field_ax, label='B poloidal (T)')
            gs_field_fig.tight_layout()
            st.pyplot(gs_field_fig)
            download_df('Download Grad–Shafranov solution grid', gs_table, 'tis_grad_shafranov_equilibrium.csv')
            st.markdown('**Profiles derived from the source functions used by the solver**')
            st.caption('These profiles use the supplied p′ and F·F′ directly; they are not refitted from the computed flux. Pressure is relative to p(edge)=0. F and q require an independently known edge F value.')
            gs_profile_table = pd.DataFrame({
                'psi_normalized_axis_to_edge': gs_profiles['psi_normalized_axis_to_edge'],
                'pressure_gradient_Pa_per_Wb': gs_profiles['pressure_gradient_Pa_per_Wb'],
                'ffprime_T2': gs_profiles['ffprime_T2'],
                'pressure_Pa_relative_to_edge': gs_profiles['pressure_Pa_relative_to_edge'],
            })
            if 'F_Tm' in gs_profiles:
                gs_profile_table['F_Tm'] = gs_profiles['F_Tm']
                gs_profile_table['q_profile'] = gs_profiles['q_profile']
                if gs_profiles['q95'] is not None:
                    st.metric('Estimated q95', f"{gs_profiles['q95']:.5g}")
                else:
                    st.info('q95 unavailable: no sufficient closed interior flux contours around the inferred magnetic axis.')
            st.dataframe(gs_profile_table, width="stretch")
            download_df('Download Grad–Shafranov equilibrium profiles', gs_profile_table, 'tis_grad_shafranov_profiles.csv')
            gs_jtor_table = pd.DataFrame({
                'Z_m': np.repeat(gs_Z, gs_R.size), 'R_m': np.tile(gs_R, gs_Z.size),
                'j_tor_A_m2': gs_profiles['j_tor_A_m2'].ravel(),
            })
            download_df('Download Grad–Shafranov toroidal current density', gs_jtor_table, 'tis_grad_shafranov_jtor.csv')
            if not gs_result['converged']:
                st.error('Iteration limit reached before the requested tolerance. Do not treat this solution as converged.')
            st.warning(gs_result['warning'])
        except Exception as exc:
            st.error(f'Grad–Shafranov reconstruction failed: {exc}')
    st.divider()
    st.markdown('### Inverse magnetic equilibrium fit (measured data)')
    st.caption('Upload a machine-specific JSON bundle containing calibrated response matrices, measured probe/flux-loop signals and uncertainties. Optional psi_profile, pressure_gradient_Pa_per_Wb, and ffprime_T2 arrays let the Grad–Shafranov solve use supplied equilibrium source profiles directly; omit all three to infer them from the fitted flux. The perimeter remains prescribed, so this generic workflow is not a machine-specific free-boundary EFIT. No example or synthetic measurement data are substituted.')
    st.code('Required: R_m, Z_m, basis_psi_Wb_rad; magnetic_probe_response_T_per_coefficient + magnetic_probe_values_T + magnetic_probe_sigma_T and/or flux_loop_response_Wb_rad_per_coefficient + flux_loop_values_Wb_rad + flux_loop_sigma_Wb_rad. Optional coil response matrices must be paired with known coil currents.', language='text')
    inverse_bundle_file = st.file_uploader('Machine-calibrated inverse reconstruction bundle (JSON)', type=['json'], key='eq_inverse_bundle')
    inverse_regularization = st.number_input('Inverse-fit regularization', min_value=0.0, value=1e-8, format='%.3g', key='eq_inverse_regularization')
    if st.button('Fit equilibrium to magnetic measurements', type='primary', key='eq_inverse_run'):
        try:
            if inverse_bundle_file is None:
                raise ValueError('Upload a machine-specific calibrated JSON bundle before running the inverse fit.')
            inverse_bundle = json.load(inverse_bundle_file)
            from tokamak_analysis.equilibrium_reconstruction import (
                reconstruct_equilibrium_inverse,
                reconstruct_equilibrium_from_reference_flux,
                derive_equilibrium_profiles_from_sources,
            )
            inverse_result = reconstruct_equilibrium_inverse(
                inverse_bundle.get('R_m'), inverse_bundle.get('Z_m'), inverse_bundle.get('basis_psi_Wb_rad'),
                magnetic_probe_response_T_per_coefficient=inverse_bundle.get('magnetic_probe_response_T_per_coefficient'),
                magnetic_probe_values_T=inverse_bundle.get('magnetic_probe_values_T'),
                magnetic_probe_sigma_T=inverse_bundle.get('magnetic_probe_sigma_T'),
                flux_loop_response_Wb_rad_per_coefficient=inverse_bundle.get('flux_loop_response_Wb_rad_per_coefficient'),
                flux_loop_values_Wb_rad=inverse_bundle.get('flux_loop_values_Wb_rad'),
                flux_loop_sigma_Wb_rad=inverse_bundle.get('flux_loop_sigma_Wb_rad'),
                coil_probe_response_T_per_A=inverse_bundle.get('coil_probe_response_T_per_A'),
                coil_probe_currents_A=inverse_bundle.get('coil_probe_currents_A'),
                coil_flux_response_Wb_rad_per_A=inverse_bundle.get('coil_flux_response_Wb_rad_per_A'),
                coil_flux_currents_A=inverse_bundle.get('coil_flux_currents_A'),
                regularization=float(inverse_regularization),
            )
            # Follow the calibrated magnetic fit with a fixed-boundary
            # Grad–Shafranov consistency solve. This is deliberately reported
            # separately from the sensor fit: it is not a free-boundary EFIT.
            inverse_gs_result = reconstruct_equilibrium_from_reference_flux(
                inverse_result['R_m'], inverse_result['Z_m'], inverse_result['psi_Wb_rad'],
                n_profile_points=32, max_iterations=500, tolerance=1e-6,
                relaxation=1.0,
                psi_profile=inverse_bundle.get('psi_profile'),
                pressure_gradient_Pa_per_Wb=inverse_bundle.get('pressure_gradient_Pa_per_Wb'),
                ffprime_T2=inverse_bundle.get('ffprime_T2'),
            )
            # Derive displayed profiles from the same source functions used by
            # the GS solve, evaluated on its refined flux map. Do not silently
            # replace supplied equilibrium profiles with a fresh inverse fit.
            inverse_source_profiles = inverse_gs_result['inferred_profiles']
            inverse_profiles = derive_equilibrium_profiles_from_sources(
                inverse_gs_result['R_m'], inverse_gs_result['Z_m'],
                inverse_gs_result['psi_Wb_rad'],
                inverse_source_profiles['psi_profile'],
                inverse_source_profiles['pressure_gradient_Pa_per_Wb'],
                inverse_source_profiles['ffprime_T2'],
                toroidal_field_boundary_Tm=inverse_bundle.get('toroidal_field_boundary_Tm'),
            )
            st.metric('Magnetic measurement count', inverse_result['measurement_count'])
            st.metric('Weighted residual RMS', f"{inverse_result['weighted_residual_rms']:.4g}")
            st.metric('Grad–Shafranov refinement', 'Converged' if inverse_gs_result['converged'] else 'Not converged')
            st.metric('GS-vs-fit flux NRMSE', f"{inverse_gs_result['reference_comparison']['interior_flux_nrmse_by_reference_range']:.4g}")
            st.caption(f"Grad–Shafranov source profiles: {inverse_gs_result['reference_comparison']['source_profile_model']}")
            inverse_refined_grid = pd.DataFrame({
                'Z_m': np.repeat(inverse_gs_result['Z_m'], inverse_gs_result['R_m'].size),
                'R_m': np.tile(inverse_gs_result['R_m'], inverse_gs_result['Z_m'].size),
                'magnetic_fit_psi_Wb_rad': inverse_gs_result['reference_psi_Wb_rad'].ravel(),
                'gs_refined_psi_Wb_rad': inverse_gs_result['psi_Wb_rad'].ravel(),
                'psi_difference_Wb_rad': (inverse_gs_result['psi_Wb_rad'] - inverse_gs_result['reference_psi_Wb_rad']).ravel(),
            })
            download_df('Download GS-refined inverse equilibrium', inverse_refined_grid,
                        'tis_gs_refined_inverse_equilibrium.csv')
            st.markdown('**Magnetic-fit flux versus Grad–Shafranov-refined flux**')
            inverse_refine_fig = Figure(figsize=(12, 4))
            inverse_refine_axes = inverse_refine_fig.subplots(1, 3)
            inverse_refine_fields = [
                (inverse_gs_result['reference_psi_Wb_rad'], 'Magnetic-measurement fit'),
                (inverse_gs_result['psi_Wb_rad'], 'GS-refined flux'),
                (inverse_gs_result['psi_Wb_rad'] - inverse_gs_result['reference_psi_Wb_rad'], 'GS refinement − fit'),
            ]
            for inverse_refine_ax, (inverse_refine_field, inverse_refine_title) in zip(inverse_refine_axes, inverse_refine_fields):
                inverse_refine_mesh = inverse_refine_ax.pcolormesh(
                    inverse_gs_result['R_m'], inverse_gs_result['Z_m'],
                    inverse_refine_field, shading='auto',
                )
                inverse_refine_ax.set_xlabel('R (m)')
                inverse_refine_ax.set_ylabel('Z (m)')
                inverse_refine_ax.set_title(inverse_refine_title)
                inverse_refine_fig.colorbar(inverse_refine_mesh, ax=inverse_refine_ax)
            inverse_refine_fig.tight_layout()
            st.pyplot(inverse_refine_fig)
            st.json({key: value for key, value in inverse_result.items() if key not in {'psi_Wb_rad', 'psi_normalized', 'boundary_mask', 'psi_standard_uncertainty_Wb_rad'}})
            st.caption(inverse_result['uncertainty_interpretation'])
            inverse_grid = pd.DataFrame({
                'Z_m': np.repeat(inverse_result['Z_m'], inverse_result['R_m'].size),
                'R_m': np.tile(inverse_result['R_m'], inverse_result['Z_m'].size),
                'psi_Wb_rad': inverse_result['psi_Wb_rad'].ravel(),
                'psi_standard_uncertainty_Wb_rad': inverse_result['psi_standard_uncertainty_Wb_rad'].ravel(),
                'psi_normalized': inverse_result['psi_normalized'].ravel(),
                'boundary_mask': inverse_result['boundary_mask'].ravel(),
            })
            st.dataframe(inverse_grid.head(30), width="stretch")
            download_df('Download inverse equilibrium grid and uncertainty', inverse_grid, 'tis_inverse_equilibrium_grid.csv')
            st.markdown('**Derived equilibrium profiles from fitted flux**')
            st.caption(
                'p′ and F·F′ are inferred from the Grad–Shafranov source and may be underdetermined. '
                'For q(ψ), the JSON bundle may provide an independently known edge F=R·Bφ value in T·m as toroidal_field_boundary_Tm.'
            )
            inv_profile_table = pd.DataFrame({
                'psi_normalized_axis_to_edge': inverse_profiles['psi_normalized_axis_to_edge'],
                'pressure_gradient_Pa_per_Wb': inverse_profiles['pressure_gradient_Pa_per_Wb'],
                'ffprime_T2': inverse_profiles['ffprime_T2'],
                'pressure_Pa_relative_to_edge': inverse_profiles['pressure_Pa_relative_to_edge'],
                'samples_in_flux_bin': inverse_profiles['profile_bin_counts'],
                'source_fit_rank': inverse_profiles['profile_bin_design_rank'],
            })
            if 'F_Tm' in inverse_profiles:
                inv_profile_table['F_Tm'] = inverse_profiles['F_Tm']
                inv_profile_table['q_profile'] = inverse_profiles['q_profile']
            st.metric('Joint p′ / FF′ fit', 'Identifiable' if inverse_profiles['joint_pressure_ffprime_identifiable'] else 'Underdetermined')
            st.dataframe(inv_profile_table, width="stretch")
            download_df('Download inverse-fit equilibrium profiles', inv_profile_table, 'tis_inverse_equilibrium_profiles.csv')
            inv_jtor_table = pd.DataFrame({
                'Z_m': np.repeat(inverse_result['Z_m'], inverse_result['R_m'].size),
                'R_m': np.tile(inverse_result['R_m'], inverse_result['Z_m'].size),
                'j_tor_A_m2': inverse_profiles['j_tor_A_m2'].ravel(),
            })
            download_df('Download inverse-fit toroidal current-density grid', inv_jtor_table,
                        'tis_inverse_toroidal_current_density.csv')
            st.json({key: value for key, value in inverse_profiles.items()
                     if key not in {'operator_Wb_rad_m-2', 'psi_normalized_grid', 'j_tor_A_m2'}})
            st.warning(inverse_result['warning'])
        except Exception as exc:
            st.error(f'Inverse magnetic reconstruction failed: {exc}')
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
                st.dataframe(out,width="stretch")
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
    workflow=st.selectbox('Select workflow',['Profile integration & gradients','Fast-ion / neutron energy spectrum','Data-driven correlation matrix','Triple-probe temperature','Thomson multi-channel spectral inference (canonical Analysis 1)','Microwave CTS spectral inversion (frequency-domain)','Thomson spatial profile reconstruction from calibrated spectra','Flux-loop magnetic-flux reconstruction','Poloidal magnetic-flux map reconstruction','Faraday-rotation polarimetry','Registered analysis modules'])
    if workflow=='Thomson multi-channel spectral inference (canonical Analysis 1)':
        st.info('Analysis 1 has one canonical execution interface. Opening it from Advanced Analysis avoids a second implementation with divergent inputs or outputs.')
        if st.button('Open canonical Thomson inference workflow',type='primary',key='open_canonical_thomson_from_advanced'):
            st.session_state['spectral_analysis_workflow']='Thomson multi-channel spectral inference'
            st.session_state['jump']='Spectral Analysis'
            st.rerun()
    elif workflow=='Microwave CTS spectral inversion (frequency-domain)':
        st.markdown('Infer plasma and ion parameters from a calibrated microwave CTS frequency-shift spectrum.')
        cts_model=st.selectbox('CTS physical forward model',[
            'electrostatic','full_electromagnetic'
        ],index=None,placeholder='Select a CTS physical model',format_func=lambda v: 'Unmagnetized electrostatic longitudinal CTS' if v=='electrostatic' else 'Magnetized full-electromagnetic CTS',key='cts_physical_model')
        if df is None or len(numeric)<2:
            st.info('Load a table with calibrated frequency-shift and spectral-power columns.')
        elif cts_model is None:
            st.info('Select the CTS physical model explicitly. No model is selected automatically.')
        elif cts_model=='electrostatic':
            st.caption('This workflow uses the unmagnetized electrostatic longitudinal dielectric model. Electron flow is explicit; one common ion temperature and projected ion flow are fitted across supplied background species.')
            c1,c2=st.columns(2)
            with c1:
                cts_frequency_col=col_select('Frequency shift (Hz)',numeric,'cts_frequency_col')
                cts_spectrum_col=col_select('Calibrated CTS spectral power', [x for x in numeric if x!=cts_frequency_col] or numeric,'cts_spectrum_col')
                cts_uncertainty_col=st.selectbox('Optional spectral 1σ uncertainty column',['<none>']+[x for x in numeric if x not in {cts_frequency_col,cts_spectrum_col}],key='cts_uncertainty_col')
                cts_probe_frequency=st.text_input('Probe frequency (Hz)',key='cts_probe_frequency')
                cts_angle=st.text_input('Scattering angle (rad)',key='cts_scattering_angle')
                cts_density_initial=st.text_input('Initial electron density (m⁻³; required optimizer start)',key='cts_density_initial')
                cts_temperature_initial=st.text_input('Initial electron temperature (eV; required optimizer start)',key='cts_temperature_initial')
                cts_ion_temperature_initial=st.text_input('Initial common ion temperature (eV; required optimizer start)',key='cts_ion_temperature_initial')
                cts_ion_flow_initial=st.text_input('Initial common ion-flow projection (m/s; required optimizer start)',key='cts_ion_flow_initial')
            with c2:
                cts_response_upload=st.file_uploader('Measured frequency-response matrix CSV',type=['csv','txt'],key='cts_response_upload')
                cts_response_unc_upload=st.file_uploader('Optional response-matrix 1σ CSV',type=['csv','txt'],key='cts_response_unc_upload')
                cts_covariance_upload=st.file_uploader('Optional frequency-bin covariance CSV',type=['csv','txt'],key='cts_covariance_upload')
                cts_method=st.selectbox('CTS inversion method',['local_least_squares','differential_evolution','improved_genetic','regularized'],index=None,placeholder='Select method explicitly',key='cts_inversion_method')
                cts_global_search=cts_method in {'differential_evolution','improved_genetic'}
                cts_parameter_bounds_json=st.text_area('Optional explicit physical parameter bounds JSON (leave blank for unbounded positive log-parameters)',placeholder='{"electron_density_m3":[1e18,1e20],"electron_temperature_eV":[50,500],"ion_temperature_eV":[10,200]}',key='cts_parameter_bounds')
                st.caption('Local least squares imposes no scientific bounds on positive log-parameters. Differential evolution and genetic search require explicit finite bounds. Regularized mode is only for an inferred tabulated fast-ion VDF.')
                cts_regularization=st.text_input('Regularization strength (required only for regularized mode)',value='',key='cts_regularization')
                cts_probe_frequency_unc=st.text_input('Probe-frequency 1σ uncertainty (Hz; optional)',key='cts_probe_frequency_unc')
                cts_angle_unc=st.text_input('Scattering-angle 1σ uncertainty (rad; optional)',key='cts_angle_unc')
                cts_electron_flow=st.text_input('Electron-flow projection (m/s; required; enter 0 only to assume stationary flow)',key='cts_electron_flow')
                cts_electron_flow_unc=st.text_input('Electron-flow projection 1σ uncertainty (m/s; optional)',key='cts_electron_flow_unc')
            st.markdown('**Ion composition (externally supplied)**')
            s1,s2,s3=st.columns(3)
            with s1: cts_species_text=st.text_input('Ion species names (comma-separated; required)',key='cts_species')
            with s2: cts_charge_text=st.text_input('Charge states (comma-separated; required)',key='cts_charges')
            with s3: cts_mass_text=st.text_input('Ion masses (amu; comma-separated; required)',key='cts_masses')
            cts_fraction_text=st.text_input('Charge-density fractions (comma-separated; required; sum to 1)',key='cts_fractions')
            cts_fraction_unc_text=st.text_input('Ion-fraction 1σ uncertainties (comma-separated; optional)',key='cts_fraction_unc')
            cts_fast_mode=st.selectbox('Fast-ion model',['None','Hot Maxwellian component','Fixed tabulated 1-D VDF','Infer regularized tabulated 1-D VDF'],key='cts_fast_mode')
            cts_fast_vdf_upload=None
            if cts_fast_mode in {'Fixed tabulated 1-D VDF','Infer regularized tabulated 1-D VDF'}:
                cts_fast_vdf_upload=st.file_uploader('Fast-ion VDF CSV (velocity_m_s, probability_density_m_s_inverse)',type=['csv','txt'],key='cts_fast_vdf')
            if cts_fast_mode!='None':
                cts_fast_species=st.text_input('Fast-ion species index (required)',value='',key='cts_fast_species')
                cts_fast_fraction=st.text_input('Initial fast-ion fraction (0–1; required)',value='',key='cts_fast_fraction')
            else:
                cts_fast_species=''; cts_fast_fraction=''
            if cts_fast_mode=='Hot Maxwellian component':
                cts_fast_temperature=st.text_input('Initial fast-ion temperature (eV; required)',value='',key='cts_fast_temperature')
                cts_fast_temperature_bounds=st.text_input('Optional fast-ion temperature bounds (eV; JSON pair)',value='',key='cts_fast_temperature_bounds')
            if cts_fast_mode=='Infer regularized tabulated 1-D VDF':
                cts_fast_smoothness=st.text_input('VDF smoothness parameter (required)',value='',key='cts_fast_smoothness')
            if st.button('Run microwave CTS inversion',type='primary',key='run_cts_inversion'):
                try:
                    frequency=pd.to_numeric(df[cts_frequency_col],errors='coerce').to_numpy(dtype=float)
                    spectrum=pd.to_numeric(df[cts_spectrum_col],errors='coerce').to_numpy(dtype=float)
                    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(spectrum)):
                        raise ValueError('CTS frequency and spectrum columns must contain only finite numeric values.')
                    if cts_uncertainty_col!='<none>':
                        cts_uncertainty=pd.to_numeric(df[cts_uncertainty_col],errors='coerce').to_numpy(dtype=float)
                        if not np.all(np.isfinite(cts_uncertainty)) or np.any(cts_uncertainty<=0):
                            raise ValueError('CTS spectral uncertainties must be finite and positive.')
                    else: cts_uncertainty=None
                    if cts_response_upload is None:
                        raise ValueError('Upload the measured frequency-response matrix.')
                    cts_response=pd.read_csv(cts_response_upload,header=None).to_numpy(dtype=float)
                    cts_spectral_cov=None if cts_covariance_upload is None else pd.read_csv(cts_covariance_upload,header=None).to_numpy(dtype=float)
                    if not cts_electron_flow.strip():
                        raise ValueError('Enter the electron-flow projection explicitly; enter 0 only if stationary electron flow is an intentional assumption.')
                    species_names=[v.strip() for v in cts_species_text.split(',') if v.strip()]
                    charges=[float(v.strip()) for v in cts_charge_text.split(',') if v.strip()]
                    masses=[float(v.strip()) for v in cts_mass_text.split(',') if v.strip()]
                    fractions=[float(v.strip()) for v in cts_fraction_text.split(',') if v.strip()]
                    if not (len(species_names)==len(charges)==len(masses)==len(fractions)):
                        raise ValueError('Ion names, charge states, masses and fractions must have equal lengths.')
                    ion_species=[{'name':name,'charge_state':charge,'mass_amu':mass} for name,charge,mass in zip(species_names,charges,masses)]
                    fast_ion=None
                    if cts_fast_mode!='None':
                        if not cts_fast_species.strip() or not cts_fast_fraction.strip():
                            raise ValueError('Enter the fast-ion species index and initial fraction explicitly.')
                        fast_species_index=int(cts_fast_species)
                        fast_fraction_initial=float(cts_fast_fraction)
                    if cts_fast_mode=='Hot Maxwellian component':
                        if not cts_fast_temperature.strip():
                            raise ValueError('Enter the fast-ion temperature initial estimate explicitly.')
                        fast_ion={'kind':'maxwellian','species_index':fast_species_index,
                            'fraction_initial':fast_fraction_initial,
                            'temperature_initial_eV':float(cts_fast_temperature)}
                        if cts_fast_temperature_bounds.strip():
                            fast_ion['temperature_bounds_eV']=json.loads(cts_fast_temperature_bounds)
                    elif cts_fast_mode in {'Fixed tabulated 1-D VDF','Infer regularized tabulated 1-D VDF'}:
                        if cts_fast_vdf_upload is None:
                            raise ValueError('Upload a two-column fast-ion VDF CSV.')
                        vdf=pd.read_csv(cts_fast_vdf_upload,header=None).to_numpy(dtype=float)
                        if vdf.ndim!=2 or vdf.shape[1]!=2:
                            raise ValueError('Fast-ion VDF CSV must contain exactly two columns: velocity and probability density.')
                        fast_ion={'kind':'tabulated_1d' if cts_fast_mode=='Fixed tabulated 1-D VDF' else 'infer_tabulated_1d',
                            'species_index':fast_species_index,'fraction_initial':fast_fraction_initial,
                            'velocity_m_s':vdf[:,0]}
                        if cts_fast_mode=='Fixed tabulated 1-D VDF':
                            fast_ion['probability_density_m_s_inverse']=vdf[:,1]
                        else:
                            if not cts_fast_smoothness.strip():
                                raise ValueError('Enter the VDF smoothness parameter explicitly.')
                            fast_ion['initial_probability_density_m_s_inverse']=vdf[:,1]
                            fast_ion['smoothness']=float(cts_fast_smoothness)
                    if cts_method is None:
                        raise ValueError('Select the CTS inversion method explicitly.')
                    if any(not value.strip() for value in (
                        cts_density_initial,cts_temperature_initial,cts_ion_temperature_initial,cts_ion_flow_initial
                    )):
                        raise ValueError('Enter all four physical optimizer initial estimates explicitly.')
                    parameter_bounds=None if not cts_parameter_bounds_json.strip() else json.loads(cts_parameter_bounds_json)
                    kwargs_cts={
                        'probe_frequency_Hz':float(cts_probe_frequency),
                        'scattering_geometry':{'scattering_angle_rad':float(cts_angle)},
                        'ion_species':ion_species,'ion_fractions':fractions,
                        'instrument_response':cts_response,'signal_uncertainty':cts_uncertainty,
                        'spectral_covariance':cts_spectral_cov,
                        'electron_density_initial_m3':float(cts_density_initial),
                        'electron_temperature_initial_eV':float(cts_temperature_initial),
                        'ion_temperature_initial_eV':float(cts_ion_temperature_initial),
                        'ion_velocity_initial_m_s':float(cts_ion_flow_initial),
                        'parameter_bounds':parameter_bounds,
                        'electron_velocity_projection_m_s':float(cts_electron_flow),
                        'fast_ion':fast_ion,'inversion_method':cts_method,
                        'global_search':bool(cts_global_search),
                    }
                    external_unc={}
                    if cts_probe_frequency_unc.strip(): external_unc['probe_frequency_Hz']=float(cts_probe_frequency_unc)
                    if cts_angle_unc.strip(): external_unc['scattering_angle_rad']=float(cts_angle_unc)
                    if cts_electron_flow_unc.strip(): external_unc['electron_velocity_projection_m_s']=float(cts_electron_flow_unc)
                    if cts_fraction_unc_text.strip():
                        fraction_unc=[float(v.strip()) for v in cts_fraction_unc_text.split(',') if v.strip()]
                        external_unc['ion_fractions']=fraction_unc
                    if external_unc: kwargs_cts['external_parameter_uncertainties']=external_unc
                    if cts_method=='regularized':
                        if not cts_regularization.strip():
                            raise ValueError('Enter the regularization strength explicitly for regularized inversion.')
                        kwargs_cts['regularization_strength']=float(cts_regularization)
                    result_cts=analyze_cts_spectrum(frequency,spectrum,**kwargs_cts)
                    st.success('CTS spectral inversion completed within the declared model scope.')
                    st.metric('Electron density',f"{result_cts['electron_density_m3']:.5g} m⁻³")
                    st.metric('Electron temperature',f"{result_cts['electron_temperature_eV']:.5g} eV")
                    st.metric('Ion temperature',f"{result_cts['ion_temperature_eV']:.5g} eV")
                    st.json(result_cts)
                    json_download('Export CTS inversion and uncertainty diagnostics',result_cts,'tis_cts_inversion.json')
                    add_report_section('Microwave CTS spectral inversion',result_cts)
                except Exception as e:
                    st.error(f'CTS inversion failed: {e}')

        else:
            st.caption('Full-EM CTS uses magnetized kinetic response, the Bindslev Maxwellian-electron vertex or an explicitly supplied relativistic electron VDF, and species-resolved ion temperatures. Optional ray-geometry JSON enables path-integrated Gaussian scattering-volume weighting.')
            c1,c2=st.columns(2)
            with c1:
                cts_frequency_col=col_select('Frequency shift (Hz)',numeric,'cts_full_frequency_col')
                cts_spectrum_col=col_select('Calibrated CTS spectral power',[x for x in numeric if x!=cts_frequency_col] or numeric,'cts_full_spectrum_col')
                cts_uncertainty_col=st.selectbox('Spectral 1σ uncertainty column (required unless covariance is supplied)',['<none>']+[x for x in numeric if x not in {cts_frequency_col,cts_spectrum_col}],key='cts_full_uncertainty_col')
                cts_probe_frequency=st.text_input('Probe frequency (Hz)',key='cts_full_probe_frequency')
                cts_density_initial=st.text_input('Initial electron density (m⁻³)',key='cts_full_density_initial')
                cts_temperature_initial=st.text_input('Initial electron temperature (eV)',key='cts_full_temperature_initial')
                cts_ion_temperature_initial=st.text_input('Initial ion temperatures (eV; one per species, comma-separated)',key='cts_full_ion_temperature_initial')
                cts_magnetic_field=st.text_input('Magnetic field vector (T; x,y,z)',key='cts_full_magnetic_field')
            with c2:
                cts_response_upload=st.file_uploader('Measured frequency-response matrix CSV',type=['csv','txt'],key='cts_full_response_upload')
                cts_covariance_upload=st.file_uploader('Optional frequency-bin covariance CSV',type=['csv','txt'],key='cts_full_covariance_upload')
                cts_geometry_upload=st.file_uploader('Optional ray-traced scattering-volume geometry JSON',type=['json'],key='cts_full_geometry_upload')
                cts_electron_flow=st.text_input('Electron flow vector (m/s; x,y,z)',key='cts_full_electron_flow')
                cts_ion_flow=st.text_area('Ion flow vectors (m/s; one x,y,z vector per species, separated by semicolons)',key='cts_full_ion_flow')
                cts_harmonic_order=st.number_input('Gyro-harmonic truncation order',min_value=2,max_value=64,value=8,step=1,key='cts_full_harmonic_order')
                cts_quadrature_order=st.number_input('Momentum/velocity quadrature order',min_value=4,max_value=64,value=12,step=1,key='cts_full_quadrature_order')
                cts_damping=st.text_input('Optional resonance damping rate (rad/s)',key='cts_full_damping')
                cts_parameter_bounds_json=st.text_area('Optional explicit physical parameter bounds JSON',placeholder='{"electron_density_m3":[1e18,1e20],"electron_temperature_eV":[50,500],"ion_temperature_eV:D+":[10,200]}',key='cts_full_parameter_bounds')
            st.markdown('**Ion composition (externally supplied)**')
            s1,s2,s3=st.columns(3)
            with s1: cts_species_text=st.text_input('Ion species names (comma-separated)',key='cts_full_species')
            with s2: cts_charge_text=st.text_input('Charge states (comma-separated)',key='cts_full_charges')
            with s3: cts_mass_text=st.text_input('Ion masses (amu; comma-separated)',key='cts_full_masses')
            cts_fraction_text=st.text_input('Charge-density fractions (comma-separated; sum to 1)',key='cts_full_fractions')
            st.markdown('**Scattering geometry and polarization**')
            g1,g2=st.columns(2)
            with g1:
                cts_incident_direction=st.text_input('Incident wave direction (x,y,z; used without ray-geometry JSON)',key='cts_full_incident_direction')
                cts_incident_polarization=st.text_input('Incident polarization (x,y,z or complex JSON vector)',key='cts_full_incident_polarization')
            with g2:
                cts_scattered_direction=st.text_input('Scattered wave direction (x,y,z; used without ray-geometry JSON)',key='cts_full_scattered_direction')
                cts_scattered_polarization=st.text_input('Scattered polarization (x,y,z or complex JSON vector)',key='cts_full_scattered_polarization')
            if cts_geometry_upload is not None:
                st.caption('Incident/scattered directions are read from the uploaded frequency-resolved geometry; the text direction fields are not required.')
            cts_electron_distribution_json=st.text_area(
                'Optional relativistic electron distribution JSON',
                placeholder='{"kind":"relativistic_maxwell_juttner"}',
                key='cts_full_electron_distribution_json',
            )
            cts_fast_ions_json=st.text_area(
                'Optional relativistic fast-ion populations JSON',
                placeholder='[{"name":"fast D+","charge_state":1,"mass_amu":2.014,"density_m3":1e17,"flow_velocity_m_s":[0,0,0],"distribution":{"kind":"relativistic_bigaussian","temperature_perpendicular_eV":2e5,"temperature_parallel_eV":1e5}}]',
                key='cts_full_fast_ions_json',
            )
            if st.button('Run magnetized full-EM CTS inversion',type='primary',key='run_full_em_cts_inversion'):
                try:
                    frequency=pd.to_numeric(df[cts_frequency_col],errors='coerce').to_numpy(dtype=float)
                    spectrum=pd.to_numeric(df[cts_spectrum_col],errors='coerce').to_numpy(dtype=float)
                    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(spectrum)):
                        raise ValueError('CTS frequency and spectrum columns must contain only finite numeric values.')
                    if cts_uncertainty_col!='<none>':
                        cts_uncertainty=pd.to_numeric(df[cts_uncertainty_col],errors='coerce').to_numpy(dtype=float)
                        if not np.all(np.isfinite(cts_uncertainty)) or np.any(cts_uncertainty<=0):
                            raise ValueError('CTS spectral uncertainties must be finite and positive.')
                    else:
                        cts_uncertainty=None
                    if cts_response_upload is None:
                        raise ValueError('Upload the measured frequency-response matrix.')
                    cts_response=pd.read_csv(cts_response_upload,header=None).to_numpy(dtype=float)
                    cts_spectral_cov=None if cts_covariance_upload is None else pd.read_csv(cts_covariance_upload,header=None).to_numpy(dtype=float)
                    if cts_uncertainty is None and cts_spectral_cov is None:
                        raise ValueError('Full-EM CTS requires either a positive spectral 1σ uncertainty column or a full frequency-bin covariance matrix.')
                    species_names=[v.strip() for v in cts_species_text.split(',') if v.strip()]
                    charges=[float(v.strip()) for v in cts_charge_text.split(',') if v.strip()]
                    masses=[float(v.strip()) for v in cts_mass_text.split(',') if v.strip()]
                    fractions=[float(v.strip()) for v in cts_fraction_text.split(',') if v.strip()]
                    if not species_names or not (len(species_names)==len(charges)==len(masses)==len(fractions)):
                        raise ValueError('Ion names, charge states, masses and fractions must have equal non-zero lengths.')
                    ion_species=[{'name':name,'charge_state':charge,'mass_amu':mass} for name,charge,mass in zip(species_names,charges,masses)]
                    ion_temperatures=_parse_ts_float_list(cts_ion_temperature_initial,'Initial ion temperatures')
                    if len(ion_temperatures)!=len(species_names):
                        raise ValueError('Enter one initial ion temperature per supplied ion species.')
                    ion_temperature_initial={name:value for name,value in zip(species_names,ion_temperatures)}
                    ion_flow_parts=[v.strip() for v in cts_ion_flow.split(';') if v.strip()]
                    if len(ion_flow_parts)!=len(species_names):
                        raise ValueError('Enter one ion flow vector per supplied ion species, separated by semicolons.')
                    ion_flows={name:_parse_ts_vector(value,f'Ion flow for {name}') for name,value in zip(species_names,ion_flow_parts)}
                    if cts_geometry_upload is not None:
                        scattering_geometry=json.load(cts_geometry_upload)
                        if not isinstance(scattering_geometry,dict) or 'frequency_resolved_geometry' not in scattering_geometry:
                            raise ValueError('Ray geometry JSON must contain frequency_resolved_geometry from the CTS ray/beam helper.')
                    else:
                        scattering_geometry={
                            'incident_direction':_parse_ts_vector(cts_incident_direction,'Incident direction'),
                            'scattered_direction':_parse_ts_vector(cts_scattered_direction,'Scattered direction'),
                        }
                    electron_distribution=None if not cts_electron_distribution_json.strip() else json.loads(cts_electron_distribution_json)
                    fast_ions=None if not cts_fast_ions_json.strip() else json.loads(cts_fast_ions_json)
                    parameter_bounds=None if not cts_parameter_bounds_json.strip() else json.loads(cts_parameter_bounds_json)
                    kwargs_cts={
                        'frequency_shift_Hz':frequency,
                        'measured_spectrum':spectrum,
                        'probe_frequency_Hz':float(cts_probe_frequency),
                        'scattering_geometry':scattering_geometry,
                        'electron_density_initial_m3':float(cts_density_initial),
                        'electron_temperature_initial_eV':float(cts_temperature_initial),
                        'ion_temperature_initial_eV':ion_temperature_initial,
                        'ion_species':ion_species,
                        'ion_fractions':fractions,
                        'magnetic_field_T':_parse_ts_vector(cts_magnetic_field,'Magnetic field'),
                        'electron_flow_velocity_m_s':_parse_ts_vector(cts_electron_flow,'Electron flow'),
                        'ion_flow_velocity_m_s':ion_flows,
                        'incident_polarization':_parse_ts_complex_vector(cts_incident_polarization,'Incident polarization'),
                        'scattered_polarization':_parse_ts_complex_vector(cts_scattered_polarization,'Scattered polarization'),
                        'instrument_response':cts_response,
                        'signal_uncertainty':cts_uncertainty,
                        'spectral_covariance':cts_spectral_cov,
                        'parameter_bounds':parameter_bounds,
                        'fast_ion_populations':fast_ions,
                        'electron_distribution':electron_distribution,
                        'harmonic_order':int(cts_harmonic_order),
                        'quadrature_order':int(cts_quadrature_order),
                    }
                    if cts_damping.strip():
                        kwargs_cts['damping_rate_rad_s']=float(cts_damping)
                    result_cts=analyze_cts_spectrum_model(model='full_electromagnetic',**kwargs_cts)
                    st.success('Magnetized full-EM CTS inversion completed within the declared model scope.')
                    st.metric('Electron density',f"{result_cts['electron_density_m3']:.5g} m⁻³")
                    st.metric('Electron temperature',f"{result_cts['electron_temperature_eV']:.5g} eV")
                    st.json(result_cts)
                    json_download('Export full-EM CTS inversion and uncertainty diagnostics',result_cts,'tis_cts_full_em_inversion.json')
                    add_report_section('Magnetized full-EM CTS inversion',result_cts)
                except Exception as e:
                    st.error(f'Full-EM CTS inversion failed: {e}')

    elif workflow=='Thomson spatial profile reconstruction from calibrated spectra':
        st.info('This subanalysis starts from measured, calibrated multi-channel Thomson spectra, infers every identifiable physical parameter with the selected forward model, and reconstructs radial profiles using PCHIP and/or a GCV smoothing spline. It does not accept pre-derived ne/Te profile points and does not extrapolate beyond measured channel coordinates.')
        if st.button('Open Thomson spectral-to-spatial-profile workflow',type='primary',key='open_thomson_profile_inference_from_advanced'):
            st.session_state['tis_thomson_profile_reconstruction_requested']=True
            st.session_state['spectral_analysis_workflow']='Thomson multi-channel spectral inference'
            st.session_state['jump']='Spectral Analysis'
            st.rerun()
    elif workflow=='Flux-loop magnetic-flux reconstruction':
        st.markdown('Reconstruct poloidal magnetic-flux change from a calibrated flux-loop voltage using Faraday’s law.')
        st.caption('Time must be in seconds and voltage in volts. Correct calibration, loop polarity, turn count, and the initial-flux condition are experiment-specific inputs.')
        st.warning('The absolute flux depends on the supplied initial flux. This workflow does not calibrate raw electronics or reconstruct Grad–Shafranov equilibrium.')
        if df is None or len(numeric) < 2:
            st.info('Load a table with time (s) and calibrated flux-loop voltage (V) columns.')
        else:
            time_col=col_select('Time (s)',numeric,'tis_flux_loop_time')
            voltage_col=col_select('Calibrated flux-loop voltage (V)',numeric,'tis_flux_loop_voltage')
            turns=int(st.number_input('Number of turns',min_value=1,value=1,step=1,key='tis_flux_loop_turns'))
            polarity=st.selectbox('Sensor polarity',options=[1,-1],format_func=lambda value: '+1' if value==1 else '−1',key='tis_flux_loop_polarity')
            offset=st.number_input('Voltage baseline offset to subtract (V)',value=0.0,format='%.8g',key='tis_flux_loop_offset')
            initial_flux=st.number_input('Initial flux at first sample (Wb)',value=0.0,format='%.8g',key='tis_flux_loop_initial')
            gap_guard=st.checkbox('Reject unexpectedly large sampling gaps',value=False,key='tis_flux_loop_gap_guard')
            max_gap=None
            if gap_guard:
                max_gap=st.number_input('Maximum allowed sample interval (s)',min_value=1e-15,value=1e-3,format='%.8g',key='tis_flux_loop_max_gap')
            if st.button('Reconstruct magnetic flux',type='primary',key='run_flux_loop_analysis'):
                try:
                    if time_col==voltage_col:
                        raise ValueError('Time and voltage must use different columns.')
                    time_values=pd.to_numeric(df[time_col],errors='raise').to_numpy(dtype=float)
                    voltage_values=pd.to_numeric(df[voltage_col],errors='raise').to_numpy(dtype=float)
                    result=analyze_flux_loop(time_values,voltage_values,turns=turns,polarity=int(polarity),
                        voltage_offset_V=float(offset),initial_flux_Wb=float(initial_flux),max_gap_s=max_gap)
                    st.success('Flux-loop analysis completed within the declared calibrated-input scope.')
                    c1,c2,c3=st.columns(3)
                    c1.metric('Flux change (final)',f"{result.flux_change_Wb[-1]:.6g} Wb")
                    c2.metric('Reconstructed flux (final)',f"{result.flux_Wb[-1]:.6g} Wb")
                    c3.metric('Samples',f"{result.diagnostics['sample_count']}")
                    frame=pd.DataFrame({'Time (s)':result.time_s,'Corrected loop voltage (V)':result.voltage_V,
                        'Integrated voltage (V s)':result.loop_voltage_integral_Vs,'Flux change (Wb)':result.flux_change_Wb,
                        'Reconstructed flux (Wb)':result.flux_Wb})
                    st.markdown('**Corrected voltage**')
                    st.line_chart(frame.set_index('Time (s)')[['Corrected loop voltage (V)']])
                    st.markdown('**Magnetic flux reconstruction**')
                    st.line_chart(frame.set_index('Time (s)')[['Flux change (Wb)','Reconstructed flux (Wb)']])
                    st.markdown('**Sampling and signal diagnostics**')
                    st.json(result.diagnostics)
                    download_df('Download flux-loop results (CSV)',frame,'tis_flux_loop_results.csv')
                    json_download('Export flux-loop result and provenance (JSON)',result.to_dict(),'tis_flux_loop_results.json')
                    with st.expander('Assumptions and limitations'):
                        st.markdown('- Faraday convention: V_loop = −N dΦ/dt after applying the selected polarity.')
                        st.markdown('- Input voltage must already be calibrated in volts; constant offset correction is optional.')
                        st.markdown('- Flux change is relative to the first sample; absolute flux uses the explicit initial condition.')
                        st.markdown('- Voltage, timing, calibration, baseline-drift and initial-condition uncertainties are not propagated.')
                        st.markdown('- This is not an equilibrium solver and does not infer plasma current or loop geometry.')
                except Exception as exc:
                    st.error(f'Flux-loop analysis failed: {exc}')
    elif workflow=='Faraday-rotation polarimetry':
        from tokamak_analysis.polarimetry import analyze_polarimetry
        st.markdown('Convert calibrated Faraday rotation to a line-integrated density-weighted parallel magnetic field.')
        st.caption('Wavelength in metres; signed rotation in radians. Optional density and path length enable a chord-averaged field only under the uniform-density assumption.')
        wavelength_text=st.text_input('Wavelength values (m; comma-separated)',value='1.06e-5',key='pol_wavelength')
        rotation_text=st.text_input('Signed Faraday rotation values (rad; comma-separated)',value='0.001',key='pol_rotation')
        density_text=st.text_input('Electron density (m⁻³; optional scalar)',value='',key='pol_density')
        path_text=st.text_input('Chord length (m; optional scalar)',value='',key='pol_path')
        rotation_unc_text=st.text_input('Rotation 1-sigma uncertainty (rad; optional scalar)',value='',key='pol_rotation_unc')
        wavelength_unc_text=st.text_input('Wavelength 1-sigma uncertainty (m; optional scalar)',value='',key='pol_wavelength_unc')
        density_unc_text=st.text_input('Density 1-sigma uncertainty (m⁻³; optional scalar)',value='',key='pol_density_unc')
        path_unc_text=st.text_input('Path-length 1-sigma uncertainty (m; optional scalar)',value='',key='pol_path_unc')
        if st.button('Analyze Faraday rotation',type='primary',key='run_polarimetry'):
            try:
                def parse_values(text):
                    return np.asarray([float(part.strip()) for part in text.split(',') if part.strip()],dtype=float)
                wavelength_values=parse_values(wavelength_text)
                rotation_values=parse_values(rotation_text)
                if wavelength_values.size!=rotation_values.size:
                    raise ValueError('Wavelength and rotation arrays must contain the same number of values.')
                optional={}
                for text_value,key in [(density_text,'electron_density_m3'),(path_text,'path_length_m'),(rotation_unc_text,'rotation_uncertainty_rad'),(wavelength_unc_text,'wavelength_uncertainty_m'),(density_unc_text,'density_uncertainty_m3'),(path_unc_text,'path_length_uncertainty_m')]:
                    if text_value.strip():
                        parsed=parse_values(text_value)
                        optional[key]=float(parsed[0]) if parsed.size==1 else parsed
                result=analyze_polarimetry(wavelength_values,rotation_values,**optional)
                st.success('Faraday-rotation polarimetry analysis completed within its declared scope.')
                st.json(_json_safe(result))
                frame=pd.DataFrame({'wavelength_m':result['wavelength_m'],'rotation_rad':result['rotation_rad'],'line_integrated_ne_B_parallel_m-2_T':result['line_integrated_ne_B_parallel_m-2_T']})
                if 'average_B_parallel_T' in result:
                    frame['average_B_parallel_T']=result['average_B_parallel_T']
                st.line_chart(frame.set_index('wavelength_m'))
                download_df('Export polarimetry results CSV',frame,'tis_polarimetry_results.csv')
                json_download('Export polarimetry result and limitations JSON',result,'tis_polarimetry_results.json')
            except Exception as exc:
                st.error(f'Faraday-rotation analysis failed: {exc}')
    elif workflow=='Poloidal magnetic-flux map reconstruction':
        st.markdown('Reconstruct relative axisymmetric poloidal flux from calibrated B_R and B_Z grids.')
        st.caption('Input table must provide calibrated B_R and B_Z grids in tesla as a long-form table with one row per (R,Z) coordinate pair. This is not EFIT or a Grad–Shafranov solver.')
        if df is None or len(numeric) < 4:
            st.info('Load a table containing major radius R (m), vertical coordinate Z (m), radial field B_R (T), and vertical field B_Z (T).')
        else:
            rcol=col_select('Major radius R (m)',numeric,'tis_psi_R')
            zcol=col_select('Vertical coordinate Z (m)',numeric,'tis_psi_Z')
            brcol=col_select('Radial field B_R (T)',numeric,'tis_psi_BR')
            bzcol=col_select('Vertical field B_Z (T)',numeric,'tis_psi_BZ')
            initial_flux=st.number_input('Reference poloidal flux (Wb)',value=0.0,format='%.8g',key='tis_psi_initial')
            if st.button('Reconstruct poloidal flux map',type='primary',key='run_poloidal_flux'):
                try:
                    if len({rcol,zcol,brcol,bzcol}) != 4:
                        raise ValueError('R, Z, B_R, and B_Z must use four distinct columns.')
                    frame=df[[rcol,zcol,brcol,bzcol]].apply(pd.to_numeric,errors='raise').dropna()
                    if frame.empty:
                        raise ValueError('No complete numeric grid rows are available.')
                    if frame.duplicated([rcol,zcol]).any():
                        raise ValueError('Each (R,Z) coordinate pair must occur exactly once; duplicate grid points were found.')
                    R=np.sort(frame[rcol].unique().astype(float))
                    Z=np.sort(frame[zcol].unique().astype(float))
                    expected=len(R)*len(Z)
                    if len(frame)!=expected:
                        raise ValueError(f'Grid is incomplete: found {len(frame)} unique coordinate pairs; expected {expected} ({len(Z)} × {len(R)}).')
                    grid_br=frame.pivot(index=zcol,columns=rcol,values=brcol).reindex(index=Z,columns=R).to_numpy(dtype=float)
                    grid_bz=frame.pivot(index=zcol,columns=rcol,values=bzcol).reindex(index=Z,columns=R).to_numpy(dtype=float)
                    result=reconstruct_poloidal_flux(R,Z,grid_br,grid_bz,initial_flux_Wb=initial_flux)
                    st.success('Poloidal-flux reconstruction completed within the declared calibrated-input scope.')
                    for label,value in [('Flux minimum (Wb)',result['diagnostics']['flux_min_Wb']),('Flux maximum (Wb)',result['diagnostics']['flux_max_Wb']),('R-derivative RMS residual',result['diagnostics']['rms_derivative_residual_R_Wb_m']),('Z-derivative RMS residual',result['diagnostics']['rms_derivative_residual_Z_Wb_m']),('Field integrability RMS residual (Wb/m²)',result['diagnostics']['rms_integrability_residual_Wb_m2'])]:
                        st.metric(label,f'{value:.6g}')
                    flux_frame=pd.DataFrame(result['poloidal_flux_Wb'],index=Z,columns=R)
                    flux_frame.index.name='Z_m'
                    flux_frame.columns.name='R_m'
                    st.dataframe(flux_frame,width="stretch")
                    st.line_chart(flux_frame.T)
                    csv_frame=flux_frame.stack().rename('poloidal_flux_Wb').reset_index()
                    download_df('Export poloidal flux map (CSV)',csv_frame,'tis_poloidal_flux.csv')
                    json_safe_result={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items()}
                    json_download('Export poloidal flux result and diagnostics (JSON)',json_safe_result,'tis_poloidal_flux.json')
                    with st.expander('Assumptions and limitations'):
                        for item in result['assumptions']:
                            st.write('- '+item)
                except Exception as exc:
                    st.error(f'Poloidal-flux reconstruction failed: {exc}')
    elif workflow=='Registered analysis modules':
        selected_family_id=st.selectbox(
            'Primary analysis (20)',
            options=[item['id'] for item in ANALYSIS_FAMILIES],
            format_func=lambda i: f"{i}. {next(item['name'] for item in ANALYSIS_FAMILIES if item['id']==i)}",
            key='tis_primary_analysis_family',
        )
        selected_family=next(item for item in ANALYSIS_FAMILIES if item['id']==selected_family_id)
        family_entries=[item for item in ANALYSES if item['id'] in selected_family['entry_ids']]
        st.caption(selected_family['description'])
        selected_id=st.selectbox(
            'Subanalysis / workflow',
            options=[item['id'] for item in family_entries],
            format_func=lambda i: f"{i}. {next(item['name'] for item in family_entries if item['id']==i)}",
            key='tis_primary_analysis_subanalysis',
        )
        entry=next(x for x in family_entries if x['id']==selected_id)
        st.caption(f"Primary analysis: {selected_family['name']} · Subanalysis: {entry['name']}")
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
                st.dataframe(pd.DataFrame(contract['input_specs']), width="stretch", hide_index=True)
            if contract.get('output_specs'):
                st.markdown('**Outputs**')
                st.dataframe(pd.DataFrame(contract['output_specs']), width="stretch", hide_index=True)
            if contract.get('workflow_modes'):
                st.markdown('**Available execution paths**')
                st.dataframe(pd.DataFrame(contract['workflow_modes']), width="stretch", hide_index=True)
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
                st.json(result)
                if selected_id == 1 and isinstance(result, dict) and {'radius_m', 'density_m3', 'temperature_eV'} <= set(result):
                    profile_frame = pd.DataFrame({'Radius (m)': result['radius_m'], 'Electron density (m^-3)': result['density_m3'], 'Electron temperature (eV)': result['temperature_eV']})
                    st.markdown('**Electron-density profile**')
                    st.line_chart(profile_frame.set_index('Radius (m)')[['Electron density (m^-3)']])
                    st.markdown('**Electron-temperature profile**')
                    st.line_chart(profile_frame.set_index('Radius (m)')[['Electron temperature (eV)']])
                    if 'density_gradient_m-4' in result:
                        st.markdown('**Radial gradients**')
                        gradients = pd.DataFrame({'Radius (m)': result['radius_m'], 'Density gradient (m^-4)': result['density_gradient_m-4'], 'Temperature gradient (eV/m)': result['temperature_gradient_eV_m-1']})
                        st.dataframe(gradients, width="stretch", hide_index=True)
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
    render_catalog_page(st, DIAGNOSTICS, FORMULAS, diagnostic_families, json_download, ANALYSES, ANALYSIS_FAMILIES)

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
    if not quality_frame.empty: st.dataframe(quality_frame,width="stretch")
    history=st.session_state.get("tis_analysis_history",[])
    record={'software':'Tokamak Insight Suite','version':__version__,'project':project,'shot_id':shot,'analyst':analyst,'input_filename':filename,'input_size_bytes':len(raw),'input_sha256':sha,'imported_at_utc':st.session_state.get('tis_imported_at_utc'),'source_manifest':st.session_state.get('tis_source_manifest'),'import_parameters':st.session_state.get('tis_import_parameters'),'analysis_history':history,'data_quality':quality,'rows':int(len(df)),'columns':[str(x) for x in df.columns],'numeric_columns':numeric,'diagnostic_count':len(DIAGNOSTICS),'formula_count':len(FORMULAS),'diagnostic_maturity_counts':maturity_counts(DIAGNOSTICS),'analysis_maturity_counts':maturity_counts(ANALYSES),'analysis_maturity_registry':[{'id':item['id'],'name':item['name'],'maturity_level':maturity_label(item)} for item in ANALYSES],'research_release_model':'Open research beta; independent scientific validation is not a prerequisite for use or contribution.','python_version':sys.version,'platform':platform.platform(),'numpy_version':np.__version__,'pandas_version':pd.__version__,'scipy_version':__import__('scipy').__version__,'status':'Software provenance and data-quality report. Researchers are responsible for confirming units, calibration, geometry, uncertainty and model assumptions for their use case.'}
    st.markdown("### Analysis history")
    if history:
        st.dataframe(pd.DataFrame([{"analysis":h["name"],"created_at_utc":h["created_at_utc"]} for h in history]),width="stretch")
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