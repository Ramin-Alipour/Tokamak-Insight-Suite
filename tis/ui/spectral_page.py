"""Modular spectroscopy and energy-spectrum analysis page."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tis.core import (
    spectral_calibration, polynomial_calibration, fit_spectral_line,
    evaluate_spectral_fit, analyze_doppler_line_end_to_end,
    doppler_temperature_with_uncertainty_eV, doppler_velocity_with_uncertainty_m_s,
    corrected_line_ratio, xray_spectrum_summary, spectrum_moments, detect_spectral_peaks, match_spectral_lines,
)


def render_spectral_page(st, df, numeric, col_select, plot_line, json_download, download_df):
    st.title('Spectroscopy, X-ray & Energy Spectra')
    if df is None:
        st.info('Load a spectrum or energy-distribution table using the left-hand panel first.')
        st.stop()
    st.caption('Input columns must be calibrated to the physical coordinate used. Background, instrument function and detector response are not inferred automatically.')
    mode=st.radio('Workflow',['End-to-end Doppler workflow','Peak detection & line identification','Optical / emission line','Wavelength calibration','Line-ratio analysis','X-ray energy spectrum','Generic energy-spectrum moments'],horizontal=True)
    if mode=='Peak detection & line identification':
        st.caption('Detect peaks from the measured spectrum, then optionally match their coordinates against a user-supplied validated line list. This workflow does not infer atomic identity from wavelength alone.')
        xcol=col_select('Wavelength / spectral coordinate',numeric,'peak_id_x')
        ycol=col_select('Measured intensity',numeric,'peak_id_y')
        c1,c2,c3=st.columns(3)
        with c1: prominence_fraction=st.number_input('Relative prominence threshold',min_value=0.0,value=0.01,step=0.01,format='%.6g',key='peak_id_prom_frac')
        with c2: absolute_prominence=st.number_input('Absolute prominence override (0 = use relative)',min_value=0.0,value=0.0,format='%.8g',key='peak_id_prom_abs')
        with c3: min_distance=st.number_input('Minimum peak spacing (same x units)',min_value=0.0,value=0.0,format='%.8g',key='peak_id_distance')
        st.caption('The coordinate and reference line list must use identical units. Correct background and instrument response before interpreting weak/overlapping lines.')
        if st.button('Detect spectral peaks',type='primary',key='peak_id_run'):
            try:
                result=detect_spectral_peaks(df[xcol],df[ycol],min_prominence=None if absolute_prominence==0 else float(absolute_prominence),prominence_fraction=float(prominence_fraction),min_distance_coordinate=float(min_distance))
                peaks=pd.DataFrame({'peak_coordinate':result['peak_coordinate'],'peak_intensity':result['peak_intensity'],'prominence':result['prominence']})
                st.metric('Detected peaks',result['n_detected_peaks'])
                st.dataframe(peaks,use_container_width=True)
                download_df('Download detected spectral peaks',peaks,'tis_detected_spectral_peaks.csv')
                json_download('Export peak-detection settings',{k:v for k,v in result.items() if not isinstance(v,np.ndarray)},'tis_spectral_peak_detection.json')
                ref_file=st.file_uploader('Optional reference line-list CSV (wavelength/species/transition)',type=['csv'],key='peak_id_reference_file')
                if ref_file is not None:
                    refs=pd.read_csv(ref_file)
                    c1,c2,c3=st.columns(3)
                    with c1: ref_x=st.selectbox('Reference coordinate column',list(refs.columns),key='peak_id_ref_x')
                    with c2: ref_species=st.selectbox('Species column (optional)',['<none>']+list(refs.columns),key='peak_id_ref_species')
                    with c3: ref_transition=st.selectbox('Transition column (optional)',['<none>']+list(refs.columns),key='peak_id_ref_transition')
                    tolerance=st.number_input('Matching tolerance (same coordinate units)',min_value=0.000001,value=0.05,format='%.8g',key='peak_id_tolerance')
                    if st.button('Match peaks to reference line list',key='peak_id_match'):
                        try:
                            matches=match_spectral_lines(result['peak_coordinate'],refs,tolerance=float(tolerance),wavelength_column=ref_x,species_column=ref_species if ref_species!='<none>' else '__no_species__',transition_column=ref_transition if ref_transition!='<none>' else '__no_transition__')
                            st.dataframe(matches,use_container_width=True)
                            download_df('Download line matches',matches,'tis_spectral_line_matches.csv')
                            if (matches['match_status']=='AMBIGUOUS_MATCH').any():
                                st.warning('Some measured peaks have multiple reference lines within the tolerance. Do not force a unique species assignment without additional spectral/atomic evidence.')
                        except Exception as exc: st.error(f'Line matching failed: {exc}')
            except Exception as exc:
                st.error(f'Spectral peak detection failed: {exc}')
    elif mode=='End-to-end Doppler workflow':
        st.caption('Unified path: raw coordinate → wavelength calibration → optional background/relative-response correction → Gaussian fit/residuals → instrument-width correction → Doppler temperature and line-of-sight velocity with stated uncertainty. Verify the atomic transition and isotope independently.')
        xcol=col_select('Raw pixel / spectral coordinate',numeric,'e2e_sp_x')
        ycol=col_select('Measured intensity',numeric,'e2e_sp_y')
        bgcol=st.selectbox('Background column (optional)',['<none>']+numeric,key='e2e_sp_bg')
        response_col=st.selectbox('Relative instrument-response column (optional)',['<none>']+numeric,key='e2e_sp_response')
        sigma_col=st.selectbox('Intensity standard-uncertainty column (optional)',['<none>']+numeric,key='e2e_sp_sigma')
        ref_pixels_text=st.text_input('Reference pixel coordinates','100,500,900',key='e2e_ref_pixels')
        ref_wavelengths_text=st.text_input('Reference wavelengths (same unit as rest wavelength)','500,550,600',key='e2e_ref_wavelengths')
        degree=st.slider('Wavelength calibration polynomial degree',1,5,1,key='e2e_cal_degree')
        try:
            ref_pixels=[float(v.strip()) for v in ref_pixels_text.split(',')]
            ref_wavelengths=[float(v.strip()) for v in ref_wavelengths_text.split(',')]
            calibration_preview=spectral_calibration(ref_pixels,ref_wavelengths,degree=degree)
            wavelength_preview=polynomial_calibration(pd.to_numeric(df[xcol],errors='coerce'),calibration_preview['coefficients_high_to_low'])
            valid_preview=wavelength_preview[np.isfinite(wavelength_preview)]
            if len(valid_preview)<7: raise ValueError('Fewer than seven finite calibrated wavelength points are available.')
            default_min=float(np.min(valid_preview)); default_max=float(np.max(valid_preview))
        except Exception as exc:
            st.error(f'Wavelength calibration preview failed: {exc}')
            st.stop()
        c1,c2=st.columns(2)
        with c1: fit_min=st.number_input('Fit minimum wavelength',value=default_min,format='%.10g',key='e2e_fit_min')
        with c2: fit_max=st.number_input('Fit maximum wavelength',value=default_max,format='%.10g',key='e2e_fit_max')
        c1,c2,c3=st.columns(3)
        with c1: rest=st.number_input('Known rest wavelength',min_value=1e-12,value=None,placeholder='Enter verified line wavelength',key='e2e_rest')
        with c2: rest_unc=st.number_input('Rest-wavelength uncertainty',min_value=0.0,value=0.0,format='%.8g',key='e2e_rest_unc')
        with c3: mass=st.number_input('Emitting-ion mass (amu)',min_value=0.0,value=0.0,key='e2e_mass')
        c1,c2,c3=st.columns(3)
        with c1: mass_unc=st.number_input('Ion-mass uncertainty (amu)',min_value=0.0,value=0.0,format='%.8g',key='e2e_mass_unc')
        with c2: inst=st.number_input('Instrument Gaussian sigma',min_value=0.0,value=0.0,format='%.8g',key='e2e_inst')
        with c3: inst_unc=st.number_input('Instrument-sigma uncertainty',min_value=0.0,value=0.0,format='%.8g',key='e2e_inst_unc')
        assume_inst=st.checkbox('Instrumental broadening is negligible (independently justified)',value=False,key='e2e_inst_negligible')
        if st.button('Run end-to-end Doppler analysis',type='primary',key='e2e_run'):
            try:
                if rest is None: raise ValueError('Enter the independently verified rest wavelength.')
                if mass<=0: raise ValueError('Enter the actual emitting-ion mass; no isotope is assumed.')
                result=analyze_doppler_line_end_to_end(
                    df[xcol],df[ycol],ref_pixels,ref_wavelengths,rest,mass,fit_min,fit_max,inst,
                    instrument_sigma_uncertainty=inst_unc,rest_wavelength_uncertainty=rest_unc,
                    ion_mass_uncertainty_amu=mass_unc,
                    background=None if bgcol=='<none>' else df[bgcol],
                    relative_response=None if response_col=='<none>' else df[response_col],
                    intensity_uncertainty=None if sigma_col=='<none>' else df[sigma_col],
                    calibration_degree=degree,assume_instrument_negligible=assume_inst)
                st.json({k:v for k,v in result.items() if not isinstance(v,np.ndarray) and k!='line_fit' and k!='wavelength_calibration'})
                st.markdown('#### Wavelength calibration')
                st.json(result['wavelength_calibration'])
                st.markdown('#### Gaussian fit and residual diagnostics')
                st.json(result['line_fit'])
                fig=Figure(figsize=(9,4)); ax=fig.subplots()
                order_fit=np.argsort(result['calibrated_wavelength'])
                ax.plot(result['calibrated_wavelength'][order_fit],result['corrected_intensity'][order_fit],label='Corrected spectrum')
                ax.plot(result['calibrated_wavelength'][order_fit],result['fitted_intensity'][order_fit],label='Gaussian fit')
                ax.set_xlabel('Wavelength'); ax.set_ylabel('Corrected intensity'); ax.legend(); st.pyplot(fig)
                residual=result['corrected_intensity']-result['fitted_intensity']
                fig=Figure(figsize=(9,3)); ax=fig.subplots()
                ax.axhline(0.0,linewidth=1); ax.plot(result['calibrated_wavelength'][order_fit],residual[order_fit],marker='.',linestyle='none')
                ax.set_xlabel('Wavelength'); ax.set_ylabel('Residual'); st.pyplot(fig)
                out=pd.DataFrame({'raw_coordinate':df[xcol],'calibrated_wavelength':result['calibrated_wavelength'],'corrected_intensity':result['corrected_intensity'],'fitted_intensity':result['fitted_intensity'],'residual':residual})
                download_df('Download end-to-end Doppler fit CSV',out,'tis_doppler_end_to_end.csv')
                json_download('Export end-to-end Doppler analysis metadata',{k:v for k,v in result.items() if not isinstance(v,np.ndarray)},'tis_doppler_end_to_end.json')
            except Exception as exc:
                st.error(f'End-to-end Doppler analysis failed: {exc}')
    elif mode=='Optical / emission line':
        xcol=col_select('Wavelength / spectral coordinate',numeric,'sp_x'); ycol=col_select('Intensity',numeric,'sp_y')
        model=st.selectbox('Line model',['Gaussian','Lorentzian','Voigt']); sigma_col=st.selectbox('Intensity uncertainty σ (optional)',['<none>']+numeric,key='sp_sigma'); xmin=st.number_input('Fit minimum',value=float(pd.to_numeric(df[xcol],errors='coerce').min())); xmax=st.number_input('Fit maximum',value=float(pd.to_numeric(df[xcol],errors='coerce').max()))
        if st.button('Fit spectral line'):
            try:
                fit_sigma=None if sigma_col=='<none>' else pd.to_numeric(df[sigma_col],errors='coerce')
                st.session_state['tis_spectral_fit']={'xcol':xcol,'ycol':ycol,'model':model,'sigma_col':sigma_col,'fit':fit_spectral_line(df[xcol],df[ycol],xmin,xmax,model,sigma_y=fit_sigma)}
            except Exception as e: st.error(str(e))
        saved=st.session_state.get('tis_spectral_fit')
        if saved and saved.get('xcol')==xcol and saved.get('ycol')==ycol and saved.get('model')==model and saved.get('sigma_col')==sigma_col:
            fit=saved['fit']; st.json(fit)
            xplot=pd.to_numeric(df[xcol],errors='coerce').to_numpy(dtype=float)
            yplot=pd.to_numeric(df[ycol],errors='coerce').to_numpy(dtype=float)
            fitmask=np.isfinite(xplot)&np.isfinite(yplot)&(xplot>=xmin)&(xplot<=xmax)
            if fitmask.sum()>=7:
                order_plot=np.argsort(xplot[fitmask])
                xx=xplot[fitmask][order_plot]; yy=yplot[fitmask][order_plot]
                predicted=evaluate_spectral_fit(xx,fit)
                fig=Figure(figsize=(9,4)); ax=fig.subplots()
                ax.plot(xx,yy,label='Observed'); ax.plot(xx,predicted,label=f"{model} fit")
                ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.legend(); st.pyplot(fig)
                fig_res=Figure(figsize=(9,3)); ax_res=fig_res.subplots()
                ax_res.axhline(0.0,linewidth=1); ax_res.plot(xx,yy-predicted,marker='.',linestyle='none')
                ax_res.set_xlabel(xcol); ax_res.set_ylabel('Residual'); ax_res.set_title('Fit residuals')
                st.pyplot(fig_res)
                st.caption(f"Fit RMSE: {fit['fit_rmse']:.5g}; residual mean: {fit['fit_residual_mean']:.5g}; reduced χ²: {fit.get('reduced_chi_square')}. Reduced χ² is interpretable only with calibrated independent uncertainties.")
            if model=='Gaussian':
                lam0=st.number_input('Known rest wavelength λ₀ (same unit as x)',min_value=1e-12,value=None,placeholder='Enter reference wavelength',key='doppler_rest_wavelength')
                sigma_lam0=st.number_input('Rest-wavelength uncertainty',min_value=0.0,value=0.0,format='%.8g')
                mass=st.number_input('Ion mass (amu; enter the actual emitting species)',min_value=0.0,value=0.0,key='doppler_mass_amu')
                sigma_mass=st.number_input('Ion-mass uncertainty (amu)',min_value=0.0,value=0.0,format='%.8g')
                inst=st.number_input('Instrument sigma (same x unit)',min_value=0.0,value=0.0,format='%.8g')
                assume_inst_negligible=st.checkbox('Assume instrumental broadening is negligible (must be independently justified)',value=False,key='doppler_inst_negligible')
                sigma_inst=st.number_input('Instrument-sigma uncertainty',min_value=0.0,value=0.0,format='%.8g')
                if st.button('Estimate Doppler quantities'):
                    if lam0 is None:
                        st.error('Enter the independently known rest wavelength; using the fitted center would force the velocity to zero.')
                    elif mass<=0:
                        st.error('Enter the actual emitting-ion mass. No isotope is assumed by default.')
                    elif inst==0 and not assume_inst_negligible:
                        st.error('Enter the calibrated instrument width, or explicitly justify the negligible-instrument-width assumption.')
                    else:
                        sig2=float(fit['sigma'])**2-inst**2
                        if sig2<=0:
                            st.error('Instrument width equals/exceeds observed Gaussian sigma.')
                        else:
                            sigma_doppler=float(np.sqrt(sig2))
                            sigma_obs_unc=float(fit.get('standard_error_sigma',0.0))
                            sigma_doppler_unc=float(np.sqrt((float(fit['sigma'])/sigma_doppler*sigma_obs_unc)**2+(inst/sigma_doppler*sigma_inst)**2))
                            thermal=doppler_temperature_with_uncertainty_eV(sigma_doppler,lam0,mass,sigma_doppler_unc,sigma_lam0,sigma_mass)
                            velocity=doppler_velocity_with_uncertainty_m_s(float(fit['center']),lam0,float(fit.get('standard_error_center',0.0)),sigma_lam0)
                            st.json({'doppler_temperature_eV':thermal['temperature_eV'],'doppler_temperature_standard_uncertainty_eV':thermal['standard_uncertainty_eV'],'relative_temperature_standard_uncertainty':thermal['relative_standard_uncertainty'],'line_of_sight_velocity_m_s':velocity['line_of_sight_velocity_m_s'],'line_of_sight_velocity_standard_uncertainty_m_s':velocity['standard_uncertainty_m_s'],'temperature_uncertainty_method':thermal['uncertainty_method'],'velocity_uncertainty_method':velocity['uncertainty_method'],'assumptions':'Gaussian Doppler broadening; Zeeman splitting, fine structure, line blends, non-Gaussian instrument response and systematic/model uncertainty are not included.'})
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

