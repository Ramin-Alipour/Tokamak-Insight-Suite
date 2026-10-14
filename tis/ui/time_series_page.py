"""Streamlit page for time-series, MHD and calibrated integration workflows.

The page receives shared UI services as arguments so its rendering can be tested
and maintained independently from the application router.
"""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tis.core import (
    resample_time_series, filter_impact_analysis, fft_spectrum,
    time_frequency_spectrogram, autocorrelation_analysis, integrate_time_series,
    prepare_time_series, butterworth_filter, cross_analysis, cross_correlation_lag,
    mode_number_from_channels, mode_number_from_phase,
)


def render_time_series_page(st, df, numeric, col_select, plot_line, json_download,
                            download_df, cached_prepare_time_series, cached_fft,
                            cached_welch):
    st.title('Time Series, Spectral & MHD Analysis')
    if df is None:
        st.info('Load a diagnostic data file using the left-hand panel before running time-series analysis.')
        st.stop()
    tcol=st.selectbox('Time column (or sample index)',['<sample index>']+numeric,key='ts_t')
    scol=col_select('Signal channel',numeric,'ts_s')
    fs0=st.number_input('Sampling frequency if no time column (Hz; required for sample-index mode)',min_value=0.0,value=0.0,format='%.8g',help='No assumed 10 kHz default: enter the actual acquisition rate when no calibrated time column is available.')
    if tcol=='<sample index>' and fs0<=0:
        st.warning('A sampling rate is required because no time column is selected. Enter the acquisition rate above before interpreting frequency axes.')
        st.stop()
    c1,c2,c3=st.columns(3)
    with c1: hp=st.number_input('High-pass cutoff (0=off)',min_value=0.0,value=0.0)
    with c2: lp=st.number_input('Low-pass cutoff (0=off)',min_value=0.0,value=0.0)
    with c3: order=st.slider('Butterworth order',1,8,3)
    resample_jitter=st.checkbox('Explicitly resample timestamp jitter to a uniform grid (refuses large gaps)',value=False,disabled=(tcol=='<sample index>'),help='Linear interpolation is not a substitute for missing samples; the routine rejects large gaps and does not downsample.')
    try:
        raw_time=None if tcol=='<sample index>' else pd.to_numeric(df[tcol],errors='coerce').to_numpy(dtype=float)
        raw_signal=pd.to_numeric(df[scol],errors='coerce').to_numpy(dtype=float)
        if resample_jitter:
            resampled=resample_time_series(raw_time,raw_signal)
            t,y,fs=cached_prepare_time_series(resampled['time_s'],resampled['values'],fs0)
            st.warning(resampled['warning'])
            st.json({k:v for k,v in resampled.items() if k not in {'time_s','values'}})
            json_download('Export resampling manifest',{k:v for k,v in resampled.items() if k not in {'time_s','values'}},'tis_resampling_manifest.json')
        else:
            t,y,fs=cached_prepare_time_series(raw_time,raw_signal,fs0)
        filter_result=filter_impact_analysis(y,fs,hp or None,lp or None,order)
        yf=filter_result['filtered_signal']
    except Exception as e:
        st.error(f'Time-series preparation failed: {e}')
        st.stop()

    x1,x2,x3=st.columns(3); x1.metric('Finite samples',len(y)); x2.metric('Sampling rate',f'{fs:.6g} Hz'); x3.metric('Nyquist',f'{fs/2:.6g} Hz')
    plot_line(t,y,'Time (s)',scol,'Raw signal'); plot_line(t,yf,'Time (s)',scol,'Filtered signal')
    st.markdown('### Filter impact audit')
    st.json({k:v for k,v in filter_result.items() if not isinstance(v,np.ndarray) and k!='filtered_signal'})
    if len(filter_result['frequency_response_Hz']):
        fig_response=Figure(figsize=(8,3)); ax_response=fig_response.subplots()
        ax_response.plot(filter_result['frequency_response_Hz'],filter_result['forward_backward_amplitude_response'])
        ax_response.set_xlabel('Frequency (Hz)'); ax_response.set_ylabel('Amplitude response |H|²')
        ax_response.set_ylim(0,1.05); ax_response.set_title('Zero-phase filter steady-state response')
        st.pyplot(fig_response)
    f_raw,amp_raw=fft_spectrum(y,fs,'hann')
    f_filt,amp_filt=fft_spectrum(yf,fs,'hann')
    fig_compare=Figure(figsize=(9,3)); ax_compare=fig_compare.subplots()
    ax_compare.plot(f_raw,amp_raw,label='Raw'); ax_compare.plot(f_filt,amp_filt,label='Filtered')
    ax_compare.set_xlabel('Frequency (Hz)'); ax_compare.set_ylabel('Amplitude'); ax_compare.legend()
    ax_compare.set_title('Raw versus filtered FFT')
    st.pyplot(fig_compare)
    json_download('Export filter impact audit',{k:v for k,v in filter_result.items() if not isinstance(v,np.ndarray) and k!='filtered_signal'},'tis_filter_impact.json')
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
            result=mode_number_from_channels(angles,np.vstack(clean),fs_common,min_coherence=mode_coh)
            st.json(result)
            json_download('Export channel-derived mode fit',result,'tis_mode_number_from_channels.json')
        except Exception as e: st.error(str(e))
    with st.expander('Advanced: fit already-extracted phases'):
        ang=st.text_input('Sensor angles θ (radians, comma-separated)','0,1.5707963,3.1415927,4.712389',key='manual_mode_angles')
        phs=st.text_input('Measured phase (radians, comma-separated)','0,1.5707963,3.1415927,4.712389',key='manual_mode_phases')
        if st.button('Fit supplied phases',key='fit_manual_mode'):
            try: st.json(mode_number_from_phase([float(z) for z in ang.split(',')],[float(z) for z in phs.split(',')]))
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
    if st.button('Integrate calibrated signal'):
        try:
            integrated=integrate_time_series(t,y,scale=int_scale,initial_value=int_initial,baseline_start_s=baseline_start,baseline_end_s=baseline_end)
            st.json({k:v for k,v in integrated.items() if k not in {'time_s','calibrated_signal','integrated_signal'}})
            plot_line(t,integrated['integrated_signal'],'Time (s)','Integrated signal','Cumulative calibrated integral')
            out=pd.DataFrame({'time_s':t,'input_signal':y,'calibrated_signal':integrated['calibrated_signal'],'integrated_signal':integrated['integrated_signal']})
            download_df('Download integrated series',out,'tis_integrated_timeseries.csv')
            json_download('Export integration manifest',{k:v for k,v in integrated.items() if k not in {'time_s','calibrated_signal','integrated_signal'}},'tis_integration_manifest.json')
        except Exception as e: st.error(str(e))

