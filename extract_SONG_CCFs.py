"""
Script to extract the SONG CCFs

The intial steps are shown in the jupyter notebook
in this folder
"""
import sys
import os
import argparse as ap
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import glob as g
iSpec_location = '/Users/jmcc/iSpec_v20161118'
sys.path.insert(0, os.path.abspath(iSpec_location))
import ispec

strongLines = iSpec_location + '/input/regions/strong_lines/absorption_lines.txt'
line_lists_parent = iSpec_location + '/input/linelists/CCF/'
telluricLines = line_lists_parent + "Synth.Tellurics.500_1100nm/mask.lst"
atomicMaskLines = {'A0': line_lists_parent + 'HARPS_SOPHIE.A0.350_1095nm/mask.lst',
                   'F0': line_lists_parent + 'HARPS_SOPHIE.F0.360_698nm/mask.lst',
                   'G2': line_lists_parent + 'HARPS_SOPHIE.G2.375_679nm/mask.lst',
                   'K0': line_lists_parent + 'HARPS_SOPHIE.K0.378_679nm/mask.lst',
                   'K5': line_lists_parent + 'HARPS_SOPHIE.K5.378_680nm/mask.lst',
                   'M5': line_lists_parent + 'HARPS_SOPHIE.M5.400_687nm/mask.lst'}
INSTRUMENT = {'RESOLUTION': 90000}

def arg_parse():
    """
    """
    p = ap.ArgumentParser()
    p.add_argument('action', choices=['orders', 'ccfs'])
    return p.parse_args()

def normaliseContinuum(spec):
    """
    Based on example.py
    normalize_whole_spectrum_strategy1_ignoring_prefixed_strong_lines function
    """
    model = 'Splines'
    degree = 2
    nknots = None
    from_resolution = INSTRUMENT['RESOLUTION']

    # continuum fit
    order = 'median+max'
    median_wave_range = 0.01
    max_wave_range = 1.0

    strong_lines = ispec.read_line_regions(strongLines)
    continuum_model = ispec.fit_continuum(spec, \
                                          from_resolution=from_resolution, \
                                          nknots=nknots, \
                                          degree=degree, \
                                          median_wave_range=median_wave_range, \
                                          max_wave_range=max_wave_range, \
                                          model=model, \
                                          order=order, \
                                          automatic_strong_line_detection=True, \
                                          strong_line_probability=0.5, \
                                          use_errors_for_fitting=True)
    # continuum normalisation
    spec_norm = ispec.normalize_spectrum(spec, \
                                         continuum_model, \
                                         consider_continuum_errors=False)
    return spec_norm

def measureRadialVelocityWithMask(spec, ccf_mask):
    """
    Radial velocity measurement using atomic line list

    Based on example.py determine_radial_velocity_with_mask() function
    """
    models, ccf = ispec.cross_correlate_with_mask(spec, \
                                                  ccf_mask, \
                                                  lower_velocity_limit=-200, \
                                                  upper_velocity_limit=200, \
                                                  velocity_step=0.50, \
                                                  mask_depth=0.01, \
                                                  fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    try:
        rv = np.round(models[0].mu(), 2) # km/s
        rv_err = np.round(models[0].emu(), 2) # km/s
    except IndexError:
        print '\n\n\nPROBLEM RV WITH MASK, SKIPPING...\n\n\n'
        return 0.0, 0.0, components, models, ccf
    return rv, rv_err, components, models, ccf

if __name__ == "__main__":
    args = arg_parse()
    data_dir = "/Users/jmcc/Dropbox/data/SONG/20180907"
    os.chdir(data_dir)
    fitsfiles = sorted(g.glob("*.fits"))

    if args.action == 'orders':
        # loop over all fits files and extract the orders and 1D spectrum
        for fitsfile in fitsfiles:
            fits_dir = fitsfile.split('.')[0]
            if not os.path.exists(fits_dir):
                os.mkdir(fits_dir)
            with fits.open(fitsfile) as ff:
                fluxes = ff[0].data[0]
                blazes = ff[0].data[2]
                waves = ff[0].data[3]

            wave_out, flux_out, error_out = [], [], []

            fig, ax = plt.subplots(4, figsize=(20, 20), sharex=True)
            o = 1
            for wave, flux, blaze in zip(waves, fluxes, blazes):
                # normalise the blaze
                blazen = blaze / np.average(blaze)
                error = np.zeros(len(wave))
                flux_corr = flux/blazen
                # keep this for the final spectrum
                wave_out.append(wave)
                flux_out.append(flux_corr)
                error_out.append(error)

                # save per order files for doing the CCF
                order_file = "{}/{}_o{:02d}.txt".format(fits_dir, fitsfile.split('.')[0], o)
                # do nan filtering on the output
                wave_filt, flux_corr_filt, error_filt = [], [], []
                for nw, nf, ne in zip(wave, flux_corr, error):
                    if nw == nw and nf == nf and ne == ne and nf != 0:
                        # also divide wave by 10 to get nm
                        wave_filt.append(nw/10.0)
                        flux_corr_filt.append(nf)
                        error_filt.append(ne)
                wave_filt = np.array(wave_filt)
                flux_corr_filt = np.array(flux_corr_filt)
                error_filt = np.array(error_filt)
                np.savetxt(order_file,
                           np.c_[wave_filt, flux_corr_filt, error_filt],
                           fmt='%.5f\t%.4f\t%.4f',
                           header='Wave_nm  Flux  Error')

                # do the plots
                _ = ax[0].plot(wave, flux, 'k-')
                _ = ax[0].set_ylabel('Flux', fontsize=18)
                _ = ax[1].plot(wave, blaze, 'r-')
                _ = ax[1].set_ylabel('Blaze Flat', fontsize=18)
                _ = ax[2].plot(wave, blazen, 'r-')
                _ = ax[2].set_ylabel('Blaze Flat (norm_avg)', fontsize=18)
                _ = ax[3].plot(wave, flux_corr, 'g-')
                _ = ax[3].set_ylabel('Flux / Blaze Flat (norm_avg)', fontsize=18)
                _ = ax[3].set_xlabel('Wavelength (Angstroms)', fontsize=18)
                o += 1

            fig.subplots_adjust(hspace=0.0)
            fig.tight_layout()
            fig.savefig('{}/{}.png'.format(fits_dir, fitsfile.split('.')[0]), dpi=400)

            # stack and sort the orders
            wave_out = np.hstack(wave_out)
            flux_out = np.hstack(flux_out)
            error_out = np.hstack(error_out)

            n = np.where(wave_out > 4700)[0]
            wave_out = wave_out[n]
            flux_out = flux_out[n]
            error_out = error_out[n]

            temp = zip(wave_out, flux_out, error_out)
            temp = sorted(temp)
            wave_out_s, flux_out_s, error_out_s = zip(*temp)

            # save the full 1D spectrum
            np.savetxt("{}/{}_1D.txt".format(fits_dir, fitsfile.split('.')[0]),
                       np.c_[wave_out_s, flux_out_s, error_out_s],
                       fmt='%.5f\t%.4f\t%.4f',
                       header='Wave_Ang  Flux  Error')
    else:
        mask_type = 'G2'
        ccf_mask = ispec.read_cross_correlation_mask(atomicMaskLines[mask_type])

        for fitsfile in fitsfiles:
            fits_dir = fitsfile.split('.')[0]
            os.chdir(fits_dir)
            orders = sorted(g.glob('*.txt'))[10:-4]
            fig, ax = plt.subplots(len(orders), figsize=(5, 10), sharex=True)
            fig_total, ax_total = plt.subplots(2, figsize=(5, 5), sharex=True)
            for i, order in enumerate(orders):
                # read in the order
                spec = ispec.read_spectrum(order)
                print('{} Loaded...'.format(order))
                spec = normaliseContinuum(spec)
                print('{} Continuum normalised...'.format(order))
                # measure the radial velocity using a atomic mask line list
                mask_rv, mask_rv_err, mask_components, mask_models, \
                mask_ccf = measureRadialVelocityWithMask(spec, ccf_mask)
                ax[i].plot(mask_ccf['x'], mask_ccf['y'], 'k-', lw=1)
                if i==0:
                    total_ccf = np.zeros(len(mask_ccf['x']))
                if mask_rv != 0.0 and mask_rv_err != 0.0:
                    total_ccf += mask_ccf['y']
            ax[len(orders)-1].set_xlabel('RV (km/s)')
            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            fig.savefig('{}_order_ccfs.png'.format(fits_dir), dpi=300)

            # fit the CCF to normalise the baseline
            n = np.where(((mask_ccf['x'] < -50) | (mask_ccf['x'] > 50)))
            coeffs = np.polyfit(mask_ccf['x'][n], total_ccf[n], 1)
            besty = np.polyval(coeffs, mask_ccf['x'])
            total_ccf_norm = total_ccf / besty

            # plot the combined CCF and the final normalised version
            ax_total[0].plot(mask_ccf['x'], total_ccf, 'k-', lw=1)
            ax_total[0].plot(mask_ccf['x'], besty, 'r--', lw=1)
            ax_total[0].set_ylabel('CCF contrast')
            ax_total[1].plot(mask_ccf['x'], total_ccf_norm, 'k-', lw=1)
            ax_total[1].set_ylabel('CCF contrast norm')
            ax_total[1].set_xlabel('RV (km/s)')
            fig_total.tight_layout()
            fig_total.subplots_adjust(hspace=0.0)
            fig_total.savefig('{}_total_ccf.png'.format(fits_dir), dpi=300)

            # output the final CCF
            np.savetxt('{}.ccf'.format(fits_dir),
                       np.c_[mask_ccf['x'], total_ccf_norm],
                       fmt='%.3f  %.5f',
                       header='RV_kms  Contrast')

            os.chdir('../')
