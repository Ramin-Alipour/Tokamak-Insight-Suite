from __future__ import annotations
import json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tis.catalog import DIAGNOSTICS, FORMULAS, diagnostic_families
from tis.data_sources import fetch_fair_mast_signal
from tis.core import (load_table, signal_quality, sha256_bytes, linear_calibration, polynomial_calibration,
    prepare_time_series, butterworth_filter, fft_spectrum, welch_psd, cross_analysis, cross_correlation_lag,
    fit_spectral_line, doppler_temperature_eV, doppler_velocity_m_s, derive_parameters,
    uncertainty_linear, regularized_tomography, spectral_calibration, probe_iv_summary,
    linear_phase_density, langmuir_iv_analysis, mode_number_from_phase, fluctuation_flux,
    exponential_heat_flux_fit, integrate_profile, spectrum_moments, neutron_rate_from_counts,
    equilibrium_grid_derivatives, xray_spectrum_summary, abel_reconstruction, axisymmetric_radiated_power, profile_gradient_analysis, triple_probe_temperature_eV, reflectometry_cutoff_density, electron_cyclotron_resonance_field, corrected_line_ratio)

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

# Sidebar: project identity and persistent data selection, modeled on the saved dashboard reference.
with st.sidebar:
    st.markdown('## ⚛️ Tokamak Insight Suite')
    st.caption('Data Analysis · Diagnostics · Plasma Physics')
    st.divider()
    project=st.text_input('Project / experiment',st.session_state.get('project','Untitled tokamak analysis'),key='project')
    shot=st.text_input('Shot / discharge ID',st.session_state.get('shot',''),key='shot')
    analyst=st.text_input('Analyst',st.session_state.get('analyst',''),key='analyst')
    st.markdown('**Load diagnostic data**')
    upload=st.file_uploader('CSV · TXT · Excel · Parquet',type=['csv','txt','xlsx','xls','parquet','pq'],key='global_upload')
    if upload is not None:
        try:
            raw=upload.getvalue(); frame=load_table(raw,upload.name)
            st.session_state['tis_raw']=raw; st.session_state['tis_filename']=upload.name
            st.session_state['tis_df']=frame; st.session_state['tis_sha']=sha256_bytes(raw)
        except Exception as e: st.error(f'Import failed: {e}')
    if 'tis_df' in st.session_state:
        st.success(f"Loaded: {st.session_state.get('tis_filename','data')}")
        if st.button('Clear loaded data',use_container_width=True):
            for k in ['tis_raw','tis_filename','tis_df','tis_sha']: st.session_state.pop(k,None)
            st.rerun()
    st.divider()
    pages=['Home','Public Data Access','Data Import & QC','Time Series & MHD','Spectral Analysis','Probe Analysis','Plasma Parameters','Equilibrium & Reconstruction','Tomography & Bolometry','Advanced Analysis','Diagnostic Catalog','Reports & Export']
    requested=st.session_state.pop('jump','Home')
    page=st.radio('WORKSPACE',pages,index=pages.index(requested) if requested in pages else 0,label_visibility='collapsed')
    st.divider()
    st.caption('Reference: tokamak.pdf · 24 pages')
    st.caption('Independent Python project · v0.3.0')

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
    fig,ax=plt.subplots(figsize=(9,3.3)); ax.plot(x,y,lw=1.0)
    if logy: ax.set_yscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
def download_df(label, data, name): st.download_button(label,data.to_csv(index=False).encode(),name,'text/csv')
def json_download(label,obj,name): st.download_button(label,json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x: np.asarray(x).tolist()).encode(),name,'application/json')
def need_data():
    if df is None:
        st.info('Load a diagnostic data file using the left-hand panel. The diagnostic library and formula registry remain available without a data file.')
        st.stop()

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
                st.session_state['tis_raw'] = mast_csv
                st.session_state['tis_filename'] = mast_filename
                st.session_state['tis_df'] = mast_frame
                st.session_state['tis_sha'] = sha256_bytes(mast_csv)
                st.session_state['tis_source_manifest'] = mast_manifest
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
    fs0=st.number_input('Sampling frequency if no time column (Hz)',min_value=1e-9,value=10000.0,format='%.8g')
    c1,c2,c3=st.columns(3)
    with c1: hp=st.number_input('High-pass cutoff (0=off)',min_value=0.0,value=0.0)
    with c2: lp=st.number_input('Low-pass cutoff (0=off)',min_value=0.0,value=0.0)
    with c3: order=st.slider('Butterworth order',1,8,3)
    try:
        t,y,fs=prepare_time_series(None if tcol=='<sample index>' else pd.to_numeric(df[tcol],errors='coerce'),pd.to_numeric(df[scol],errors='coerce'),fs0)
        yf=butterworth_filter(y,fs,hp or None,lp or None,order)
        x1,x2,x3=st.columns(3); x1.metric('Finite samples',len(y)); x2.metric('Sampling rate',f'{fs:.6g} Hz'); x3.metric('Nyquist',f'{fs/2:.6g} Hz')
        plot_line(t,y,'Time (s)',scol,'Raw signal'); plot_line(t,yf,'Time (s)',scol,'Filtered signal')
        f,amp=fft_spectrum(yf,fs); fp,psd=welch_psd(yf,fs)
        plot_line(f,amp,'Frequency (Hz)','Amplitude','FFT amplitude'); plot_line(fp,np.maximum(psd,np.finfo(float).tiny),'Frequency (Hz)','PSD (unit²/Hz)','Welch PSD',True)
        download_df('Download filtered series',pd.DataFrame({'time_s':t,'raw':y,'filtered':yf}),'tis_timeseries.csv')
        download_df('Download FFT',pd.DataFrame({'frequency_Hz':f,'amplitude':amp}),'tis_fft.csv'); download_df('Download PSD',pd.DataFrame({'frequency_Hz':fp,'psd':psd}),'tis_welch_psd.csv')
        st.markdown('### Cross-channel coherence / phase / delay')
        others=[c for c in numeric if c!=scol]
        if others:
            other=st.selectbox('Second synchronized channel',others,key='cross_other')
            _,y2,fs2=prepare_time_series(None if tcol=='<sample index>' else pd.to_numeric(df[tcol],errors='coerce'),pd.to_numeric(df[other],errors='coerce'),fs0)
            if len(y2)==len(y) and np.isclose(fs,fs2):
                xa=cross_analysis(yf,y2,fs); lags,corr,lag=cross_correlation_lag(yf,y2,fs)
                st.metric('Peak absolute-correlation lag',f'{lag:.6g} s'); plot_line(xa['frequency_Hz'],xa['coherence'],'Frequency (Hz)','Coherence','Magnitude-squared coherence'); plot_line(xa['frequency_Hz'],xa['cross_phase_rad'],'Frequency (Hz)','Phase (rad)','Cross phase'); plot_line(lags,corr,'Lag (s)','Correlation','Cross-correlation')
                download_df('Download cross-channel results',pd.DataFrame(xa),'tis_cross_analysis.csv')
            else: st.warning('Channels need aligned timestamps and compatible sampling rates.')
        st.markdown('### Mode number from multi-sensor phase')
        st.caption('Enter measured sensor angles and synchronized phase estimates; unwrap convention and geometry must be physically consistent.')
        ang=st.text_input('Sensor angles θ (radians, comma-separated)','0,1.5707963,3.1415927,4.712389')
        phs=st.text_input('Measured phase (radians, comma-separated)','0,1.5707963,3.1415927,4.712389')
        if st.button('Fit mode number'):
            try: st.json(mode_number_from_phase([float(z) for z in ang.split(',')],[float(z) for z in phs.split(',')]))
            except Exception as e: st.error(str(e))
    except Exception as e: st.error(str(e))

elif page=='Spectral Analysis':
    st.title('Spectroscopy, X-ray & Energy Spectra'); need_data()
    st.caption('Input columns must be calibrated to the physical coordinate used. Background, instrument function and detector response are not inferred automatically.')
    mode=st.radio('Workflow',['Optical / emission line','Wavelength calibration','Line-ratio analysis','X-ray energy spectrum','Generic energy-spectrum moments'],horizontal=True)
    if mode=='Optical / emission line':
        xcol=col_select('Wavelength / spectral coordinate',numeric,'sp_x'); ycol=col_select('Intensity',numeric,'sp_y')
        model=st.selectbox('Line model',['Gaussian','Lorentzian','Voigt']); xmin=st.number_input('Fit minimum',value=float(pd.to_numeric(df[xcol],errors='coerce').min())); xmax=st.number_input('Fit maximum',value=float(pd.to_numeric(df[xcol],errors='coerce').max()))
        if st.button('Fit spectral line'):
            try:
                st.session_state['tis_spectral_fit']={'xcol':xcol,'ycol':ycol,'model':model,'fit':fit_spectral_line(df[xcol],df[ycol],xmin,xmax,model)}
            except Exception as e: st.error(str(e))
        saved=st.session_state.get('tis_spectral_fit')
        if saved and saved.get('xcol')==xcol and saved.get('ycol')==ycol and saved.get('model')==model:
            fit=saved['fit']; st.json(fit)
            if model=='Gaussian':
                lam0=st.number_input('Rest wavelength λ₀ (same unit as x)',min_value=1e-12,value=max(float(fit['center']),1e-12)); mass=st.number_input('Ion mass (amu)',min_value=1e-6,value=40.0); inst=st.number_input('Instrument sigma (same x unit)',min_value=0.0,value=0.0,format='%.8g')
                if st.button('Estimate Doppler quantities'):
                    sig2=float(fit['sigma'])**2-inst**2
                    if sig2<=0: st.error('Instrument width equals/exceeds observed Gaussian sigma.')
                    else: st.json({'doppler_temperature_eV':doppler_temperature_eV(np.sqrt(sig2),lam0,mass),'line_of_sight_velocity_m_s':doppler_velocity_m_s(float(fit['center'])-lam0,lam0),'assumptions':'Isolated line, Gaussian Doppler broadening, valid instrument-width subtraction.'})
            plot_line(np.asarray(df[xcol],float),np.asarray(df[ycol],float),xcol,ycol,'Input spectrum')
            json_download('Export fit JSON',fit,'tis_spectral_fit.json')
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
        ecol=col_select('Energy coordinate (keV)',numeric,'energy_x'); ccol=col_select('Counts / intensity',numeric,'energy_y'); bgcol=st.selectbox('Background column (optional)',['<none>']+numeric,key='energy_bg')
        if st.button('Analyze energy spectrum'):
            try:
                result=xray_spectrum_summary(df[ecol],df[ccol],None if bgcol=='<none>' else df[bgcol]) if mode=='X-ray energy spectrum' else spectrum_moments(df[ecol],df[ccol],None if bgcol=='<none>' else df[bgcol]); st.json(result); json_download('Export spectrum summary',result,'tis_energy_spectrum_summary.json')
                plot_line(df[ecol],df[ccol],ecol,ccol,'Energy spectrum')
            except Exception as e: st.error(str(e))

elif page=='Probe Analysis':
    st.title('Langmuir, Ball-Pen & Probe Analysis'); need_data()
    vcol=col_select('Probe voltage (V)',numeric,'probe_v'); icol=col_select('Probe current (A)',numeric,'probe_i')
    vmin=float(pd.to_numeric(df[vcol],errors='coerce').min()); vmax=float(pd.to_numeric(df[vcol],errors='coerce').max())
    st.markdown('### Fit-window and probe geometry')
    c1,c2,c3=st.columns(3)
    with c1: efit=st.slider('Electron-retarding fit range (V)',min_value=float(vmin),max_value=float(vmax),value=(float(vmin+(vmax-vmin)*.35),float(vmin+(vmax-vmin)*.75)))
    with c2: ifit=st.slider('Ion-saturation range (V)',min_value=float(vmin),max_value=float(vmax),value=(float(vmin),float(vmin+(vmax-vmin)*.2)))
    with c3: area=st.number_input('Effective collection area (m²; 0=omit density)',min_value=0.0,value=0.0,format='%.8g')
    c1,c2,c3=st.columns(3)
    with c1: mass=st.number_input('Main ion mass (amu)',min_value=0.0,value=2.0)
    with c2: ti=st.number_input('Ion temperature (eV; model input)',min_value=0.0,value=0.0)
    with c3: sheath=st.number_input('Sheath coefficient',min_value=0.01,value=.61)
    if st.button('Analyze probe I–V',type='primary'):
        try:
            result=langmuir_iv_analysis(df[vcol],df[icol],efit,ifit,area or None,mass or None,ti,sheath)
            st.json(result); plot_line(df[vcol],df[icol],vcol,icol,'Measured probe I–V'); json_download('Export probe analysis JSON',result,'tis_probe_analysis.json')
        except Exception as e: st.error(str(e))
    st.warning('The automated fit is a first-pass estimate. Confirm sweep polarity, fit windows, saturation regions, magnetic-field effects and the probe/sheath model before interpreting Te or density.')

elif page=='Plasma Parameters':
    st.title('Plasma Parameters & Diagnostic Conversions')
    tabs=st.tabs(['Derived parameters','Interferometry','Reflectometry / ECE','Turbulent flux','Neutron rate'])
    with tabs[0]:
        a,b,c=st.columns(3)
        with a: ne=st.number_input('Electron density nₑ (m⁻³)',min_value=0.0,value=1e19,format='%.6g'); ni=st.number_input('Ion density nᵢ (m⁻³; 0=quasineutral estimate)',min_value=0.0,value=0.0,format='%.6g'); te=st.number_input('Electron temperature Tₑ (eV)',min_value=0.0,value=1000.0); ti=st.number_input('Ion temperature Tᵢ (eV)',min_value=0.0,value=800.0)
        with b: ip=st.number_input('Plasma current Iₚ (A)',min_value=0.0,value=50000.0); vl=st.number_input('Loop voltage (V)',value=1.0); R=st.number_input('Major radius R (m)',min_value=1e-9,value=.68); minor=st.number_input('Minor radius a (m)',min_value=1e-9,value=.12)
        with c: bt=st.number_input('Toroidal field Bₜ (T)',min_value=0.0,value=1.0); mi=st.number_input('Main ion mass (amu)',min_value=1e-6,value=2.0); W=st.number_input('Stored thermal energy (J; 0=omit)',min_value=0.0,value=0.0); ploss=st.number_input('Loss power (W; 0=omit)',min_value=0.0,value=0.0)
        try: result=derive_parameters(ne,te,ti,ip,vl,R,minor,bt,W or None,ploss or None,ni or None,mi); st.json(result); json_download('Export parameter calculations',result,'tis_plasma_parameters.json')
        except Exception as e: st.error(str(e))
    with tabs[1]:
        ph=st.number_input('Interferometer phase shift (rad)',value=.1); freq=st.number_input('Probe frequency (Hz)',min_value=1.0,value=140e9,format='%.8g')
        if st.button('Convert phase to line-integrated density'): st.metric('Approximate line density (m⁻²)',f'{float(linear_phase_density(ph,freq)):.7g}')
        st.caption('Single line-integrated estimate only; not a radial profile reconstruction.')
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
    tomo_mode=st.radio('Reconstruction workflow',['General response-matrix inversion','Axisymmetric Abel inversion','Axisymmetric radiated-power integration'],horizontal=True)
    lam=st.number_input('Tikhonov regularization λ',min_value=0.0,value=.01,format='%.8g',key='tomo_lam'); nonneg=st.checkbox('Constrain reconstructed profile to be nonnegative',value=True,key='tomo_nonneg')
    if tomo_mode=='General response-matrix inversion':
        st.write('Solve y = A x + noise using a calibrated geometry / response matrix.')
        Afile=st.file_uploader('Response / geometry matrix A (CSV, no header)',type=['csv'],key='tomo_A'); yfile=st.file_uploader('Measurements y (CSV, one column)',type=['csv'],key='tomo_y')
        if Afile and yfile and st.button('Run regularized reconstruction',type='primary'):
            try:
                A=pd.read_csv(Afile,header=None).to_numpy(float); y=pd.read_csv(yfile,header=None).to_numpy(float).ravel(); r=regularized_tomography(A,y,lam,nonneg)
                st.json({k:v for k,v in r.items() if k!='solution'}); out=pd.DataFrame({'basis_or_voxel_index':np.arange(len(r['solution'])),'reconstructed_value':r['solution']}); st.dataframe(out); download_df('Download reconstruction CSV',out,'tis_tomography_solution.csv')
            except Exception as e: st.error(str(e))
    elif tomo_mode=='Axisymmetric Abel inversion':
        btxt=st.text_input('Chord impact parameters b (m, comma-separated)','0,0.02,0.04,0.06,0.08'); ytxt=st.text_input('Line-integrated measurements (units·m, comma-separated)','1.0,0.9,0.7,0.4,0.1'); etxt=st.text_input('Radial bin edges (m, comma-separated)','0,0.02,0.04,0.06,0.08,0.10')
        if st.button('Run Abel inversion',type='primary'):
            try:
                r=abel_reconstruction([float(z) for z in btxt.split(',')],[float(z) for z in ytxt.split(',')],[float(z) for z in etxt.split(',')],lam,nonneg); out=pd.DataFrame({'radius_m':r['radius_centers_m'],'reconstructed_profile':r['emissivity_or_density_profile']}); st.json({k:v for k,v in r.items() if k not in ['response_matrix','radius_centers_m','emissivity_or_density_profile']}); st.dataframe(out); plot_line(out['radius_m'],out['reconstructed_profile'],'Radius (m)','Reconstructed value','Abel inversion'); download_df('Download Abel profile',out,'tis_abel_profile.csv')
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
        if st.button('Fit exponential heat-flux profile'):
            try: r=exponential_heat_flux_fit(df[xcol],df[qcol]); st.json(r); json_download('Export heat-flux fit',r,'tis_heat_flux_fit.json')
            except Exception as e: st.error(str(e))

elif page=='Advanced Analysis':
    st.title('Advanced Diagnostic Analysis')
    workflow=st.selectbox('Select workflow',['Calibration & uncertainty','Profile integration & gradients','Fast-ion / neutron energy spectrum','Data-driven correlation matrix','Triple-probe temperature'])
    if workflow=='Calibration & uncertainty':
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
        need_data(); e=col_select('Energy coordinate (eV)',numeric,'adv_energy'); c=col_select('Counts / distribution',numeric,'adv_counts'); bg=st.selectbox('Background (optional)',['<none>']+numeric,key='adv_bg')
        if st.button('Calculate spectrum moments'):
            try: r=spectrum_moments(df[e],df[c],None if bg=='<none>' else df[bg]); st.json(r); json_download('Export energy moments',r,'tis_energy_moments.json')
            except Exception as ex: st.error(str(ex))
    elif workflow=='Triple-probe temperature':
        vd=st.number_input('Measured triple-probe voltage difference (V)',min_value=0.000001,value=10.0); mf=st.number_input('Model correction factor',min_value=0.000001,value=1.0)
        if st.button('Estimate triple-probe Te'):
            try: r=triple_probe_temperature_eV(vd,mf); st.json(r); json_download('Export triple-probe result',r,'tis_triple_probe.json')
            except Exception as e: st.error(str(e))
    else:
        need_data(); chosen=st.multiselect('Select synchronized numeric channels',numeric,default=numeric[:min(6,len(numeric))])
        if len(chosen)>=2:
            mat=df[chosen].apply(pd.to_numeric,errors='coerce').corr(); st.dataframe(mat.style.format('{:.3f}')); plot_line(np.arange(len(mat.columns)),mat.iloc[0].values,'Channel index',f'Correlation with {mat.columns[0]}','Channel correlation row'); download_df('Download correlation matrix',mat.reset_index(names='channel'),'tis_correlation_matrix.csv')

elif page=='Diagnostic Catalog':
    st.title('Diagnostic Knowledge Base')
    view=st.radio('Catalog',['Diagnostic families','Formula reference'],horizontal=True)
    query=st.text_input('Search by diagnostic, observable, formula, method or caveat','').strip().lower()
    if view=='Diagnostic families':
        fam=st.selectbox('Diagnostic family',['All families']+diagnostic_families())
        entries=[d for d in DIAGNOSTICS if (fam=='All families' or d['family']==fam) and (not query or query in ' '.join(str(v) for v in d.values()).lower())]
        st.write(f'{len(entries)} entries')
        for d in entries:
            with st.expander(f"{d['diagnostic']} · {d['family']}"):
                st.markdown(f"**Observables:** {d['quantities']}\n\n**Quantity class (M=measured, D=derived, I=inferred):** {d['status']}\n\n**Method:** {d['method']}\n\n**Limitations:** {d['caveat']}")
        json_download('Export diagnostic registry JSON',entries,'tis_diagnostic_registry.json')
    else:
        entries=[f for f in FORMULAS if not query or query in ' '.join(str(v) for v in f.values()).lower()]
        st.write(f'{len(entries)} formula entries')
        for f in entries:
            with st.expander(f['name']): st.markdown(f"**Equation:** `{f['equation']}`\n\n**Inputs:** {f['inputs']}\n\n**Output:** {f['output']}\n\n**Status:** {f['status']}\n\n**Assumptions:** {f['assumptions']}")
        json_download('Export formula registry JSON',entries,'tis_formula_registry.json')

elif page=='Reports & Export':
    st.title('Reports, Provenance & Export')
    if df is None: st.info('Load a file to create a provenance record.'); st.stop()
    record={'software':'Tokamak Insight Suite','version':'0.3.0','project':project,'shot_id':shot,'analyst':analyst,'input_filename':filename,'input_size_bytes':len(raw),'input_sha256':sha,'imported_at_utc':datetime.now(timezone.utc).isoformat(),'rows':int(len(df)),'columns':[str(x) for x in df.columns],'numeric_columns':numeric,'reference_document':'tokamak.pdf','diagnostic_count':len(DIAGNOSTICS),'formula_count':len(FORMULAS),'python_version':sys.version,'platform':platform.platform(),'status':'Analysis record; scientific validity depends on calibration, geometry, uncertainty and method assumptions.'}
    st.json(record); json_download('Download provenance JSON',record,'tis_provenance.json'); download_df('Export imported data CSV',df,'tis_imported_data.csv')
    st.markdown('### Session activity')
    st.write(st.session_state.get('tis_activity',[]))

# Activity tracking, bounded to avoid unbounded session growth.
activity=st.session_state.get('tis_activity',[])
entry=f'{page} opened · {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}'
if not activity or activity[0].split(' · ')[0]!=page: st.session_state['tis_activity']=([entry]+activity)[:10]
st.divider()
st.caption('Tokamak Insight Suite v0.3.0 · PDF-derived knowledge base and analysis workflows. Every output must be interpreted using the actual device calibration, units, geometry and uncertainty model. Specialized equilibrium solvers and collisional-radiative atomic models remain distinct modules, not implied by the catalog.')
