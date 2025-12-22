# Define the class of telescope hardware
import csv
from curses.ascii import alt
import os, sys
import numpy as np
import astropy.time
from astropy.time import Time
import pandas as pd
from astropy.coordinates import SkyCoord, AltAz, Angle
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.coordinates import get_body
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
import cv2
from astroquery.gaia import Gaia
from astropy.modeling.models import Moffat2D
import batman
from rasterio import band
import sncosmo
import phoebe
from scipy import special, constants
from scipy.linalg import solve_banded
from scipy.stats import binned_statistic
from datetime import datetime
import requests
import lightkurve as lk
from astroquery.vizier import Vizier
import galsim
from skyfield.api import load
from skyfield.iokit import parse_tle_file
import matplotlib.pyplot as plt
from scipy.special import jv
from io import StringIO
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
from astropy.timeseries import LombScargle
from pathlib import Path
from scipy.spatial import KDTree
from local_catalog import chunked_star_search, streaming_star_search
from scipy import fft
import json
from tqdm import tqdm
import time as sys_time

class hardware:
    def __init__(self,hardware_type,name = 'simulator'):
        self.connected = False
        self.name = name
        self.hardware_type = hardware_type
        self.telescope = None  # set in connect
        self._status = 'idle'
    @property
    def status(self):
        if "simulator" in self.name:            # Calculate the status according to calculation
            if self.telescope.world.time > self.idle_time:
                self._status = 'idle'
            return self._status
        else:                                   # read the status from instrumentation
            pass
    def connect(self,telescope):
        if 'simulator' in self.name:
            self.connected = True
            self.telescope = telescope
            print(self.hardware_type,self.name,'Connected to the telescope')
            for i in self.initpar:    # use set not get
                self.set_parameter(i,self.initpar[i])   # self.initpar: dictionary   i: parameter name   self.initpar[i]: value
    def set_parameter(self,attribute,value):
        if self.connected:
            if 'simulator' in self.name:
                setattr(self,attribute,value)
            else:
                setattr(self,attribute,value)
                # SET according to the hardware
                pass
        else:
            raise   RuntimeError('Please connect the hardware')

    def get_parameter(self,attribute):   # 'get' is different from 'set'
        if hasattr(self,attribute):   # Determines whether the self has the attribute.
            if 'simulator' in self.name:
                return attribute
            else:
                # read according to the hardware update the attribute in class
                pass
        else:
            raise   RuntimeError('Please connect the hardware')  
# These two classes would return the status of the instruments.
        
# pixel_size_m, pixel_number, readout_time_s, gain, dark_current_e_s_1, full_well_capacity_e, read_noise_e     
class camera(hardware):
    def __init__(self,par_camera,name = 'simulator_cam'):
        super().__init__('camera',name = name)
        self._status = 'idle'
        self.initpar = par_camera  # initpar: dictionary
        self.idle_time = -1
    def capture(self,exposure_time):
        if self.status == 'busy':
            print('Camera is busy')
            return 0
        if 'simulator' in self.name:
            self._status = 'busy'
            self.exposure_s = exposure_time
            self.idle_time = self.telescope.world.time + np.max([exposure_time,self.readout_time_s])/24/3600
            print('Capturing until',self.idle_time)

    def download(self,generated_img = -1):
        if self.status != 'idle':
            print('Camera is busy')
            return 0
        if 'simulator' in self.name:
            # print('Image is loaded')
            return generated_img

# tracking_mode, tracking_speed_deg_s_1, stable_time_s, goto_error_arcsec, tracking_error_arcsec_min_1
class mount(hardware):
    def __init__(self,par_mount,name = 'simulator_mount'):
        super().__init__('mount',name = name)  # Returns the parent class (parameters of parent class _int_)
        self._status = 'idle'                # status = idle/running/goto
        self.initpar = par_mount
        self.ra_deg = 0
        self.dec_deg = 0
        self.idle_time = -1
    def goto(self,ra_deg,dec_deg):                      # goto an ra,dec position
        if self.status == 'busy':
            print('Mount is busy')
            return 0
        if 'simulator' in self.name:
            if self.tracking_mode == 'alt-az':
                self._status = 'busy'
                dra = self.ra_deg-ra_deg-((self.ra_deg-ra_deg)//360)*360
                dra = np.min([dra,360-dra])*np.cos(np.min(np.abs([self.dec_deg,dec_deg]))/180*np.pi)
                ddec = np.abs(self.dec_deg-dec_deg)
                time4goto = (np.sqrt(dra**2+ddec**2))
                self.idle_time = self.telescope.world.time +(self.stable_time_s+ time4goto/self.tracking_speed_deg_s_1)/24/3600
                print('Mount is moving to',ra_deg,dec_deg,'until',self.idle_time,'Using',self.stable_time_s+time4goto/self.tracking_speed_deg_s_1,'s')
                self.ra_deg = ra_deg + np.random.randn()*self.goto_error_arcsec/3600
                self.dec_deg = dec_deg + np.random.randn()*self.goto_error_arcsec/3600

                
#parameter of telescope = position, seeing, focus, diameter
class telescope:
    def __init__(self,mount,camera,par_telescope ,name = 'simulator_telescope'):
        self.name = name
        self.mount = mount
        self.camera = camera
        
        for i in par_telescope:
            setattr(self,i,par_telescope[i])
        self.mount.connect(self)
        self.camera.connect(self)
        self.arcsec_pixel_1 = self.camera.pixel_size_m/self.focal_length_m * 206265 #arcsec
        self.fov_x = self.camera.pixel_number_x * self.arcsec_pixel_1/3600   #degree
        self.fov_y = self.camera.pixel_number_y * self.arcsec_pixel_1/3600
        self.fov_diag = np.sqrt(self.fov_x**2+self.fov_y**2)
    @property
    def position(self):
        if not hasattr(self, '_position'):
            self._position = EarthLocation.from_geodetic(lat=self.latlonalt[0]*u.deg, lon=self.latlonalt[1]*u.deg, height=self.latlonalt[2]*u.m)
        return self._position
    @property
    def Altaz_obs(self):
        return AltAz(obstime=self.world.time_astropy, location=self.position)
    @property
    def pointing(self):
        return SkyCoord(ra=self.mount.ra_deg*u.deg, dec=self.mount.dec_deg*u.deg, obstime=self.world.time_astropy, frame='icrs', location=self.world.telescope.position)
    
    @property
    def wcs(self):
        theta = self.rot_deg/180*np.pi
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.camera.pixel_number_x//2, self.camera.pixel_number_y//2]
        w.wcs.cdelt = np.array([self.arcsec_pixel_1/3600, self.arcsec_pixel_1/3600])
        w.wcs.crval = [self.mount.ra_deg, self.mount.dec_deg]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.pc = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        return w
    
    def wcs_to_target(self,ra_target,dec_target):
        ra_rad = self.mount.ra_deg * np.pi / 180
        dec_rad = self.mount.dec_deg * np.pi / 180
        ra_target_rad = ra_target * np.pi / 180
        dec_target_rad = dec_target * np.pi / 180
        pos_self = np.array([np.cos(dec_rad)*np.cos(ra_rad),np.cos(dec_rad)*np.sin(ra_rad),np.sin(dec_rad)])
        R_z_RA1 = np.array([[np.cos(ra_target_rad),np.sin(ra_target_rad),0],[-np.sin(ra_target_rad),np.cos(ra_target_rad),0],[0,0,1]])
        R_y_dec1 = np.array([[np.cos(dec_target_rad),0,-np.sin(dec_target_rad)],[0,1,0],[np.sin(dec_target_rad),0,np.cos(dec_target_rad)]])
        pos_target = np.dot(np.dot(R_z_RA1,R_y_dec1),pos_self)
        ra_new = np.arctan2(pos_target[1],pos_target[0])
        dec_new = np.arcsin(pos_target[2])
        theta_new = np.arccos(pos_target[2])
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.camera.pixel_number_x//2, self.camera.pixel_number_y//2]
        w.wcs.cdelt = np.array([self.arcsec_pixel_1/3600, self.arcsec_pixel_1/3600])
        w.wcs.crval = [ra_new * 180 / np.pi, dec_new * 180 / np.pi]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.pc = np.array([[np.cos(theta_new),-np.sin(theta_new)],[np.sin(theta_new),np.cos(theta_new)]])
        return w
    def build(self,world):
        self.world = world
        print("Telecope is built in world",world.name)


#parameter of world = t0
class world:
    def __init__(self,telescope,photon,sim,output,t0,name = 'simulator_world',input_schedule = None,input_event = None):
        self.name = name
        self.telescope = telescope
        self.telescope.build(self)
        self.photon = photon
        self.photon.connect(self)
        self.output = output
        if 'simulator' in self.name:
            self.img_simulator = sim
            self.img_simulator.connect(self)
        self._time = t0
        if input_schedule is not None:
            self.schedule = pd.read_csv(input_schedule)
        self.events_world = input_event

    @property
    def time(self):
        if 'simulator' in self.name:
            return self._time
        else:
            self._time = Time.now().jd
            return self._time
    @property
    def time_astropy(self):
        return Time(self.time,scale = 'utc',format = 'jd',location = self.telescope.position)
    
    def create_fits_header(self, r):
        header = fits.Header()
        # basci FITS header keywords
        header['SIMPLE'] = (True, 'file does conform to FITS standard')
        header['BITPIX'] = (-32, '8 unsigned int, 16 & 32 int, -32 & -64 real')
        header['NAXIS'] = (2, 'number of axes')
        header['NAXIS1'] = (self.telescope.camera.pixel_number_x, 'fastest changing axis')
        header['NAXIS2'] = (self.telescope.camera.pixel_number_y, 'next to fastest changing axis')

        header['BSCALE'] = (1.0, 'physical = BZERO + BSCALE*array_value')
        header['BZERO'] = (0.0, 'physical = BZERO + BSCALE*array_value')

        # time
        time_obj = self.time_astropy
        header['DATE-OBS'] = (time_obj.iso, 'YYYY-MM-DDThh:mm:ss observation start, UT')
        header['EXPTIME'] = (r['exposure_s'], 'Exposure time in seconds')
        header['EXPOSURE'] = (r['exposure_s'], 'Exposure time in seconds')

        # CCD temperature
        temp_celsius = self.telescope.camera.CCD_temperature - 273.15  
        header['SET-TEMP'] = (temp_celsius, 'CCD temperature setpoint in C')
        header['CCD-TEMP'] = (temp_celsius, 'CCD temperature at start of exposure in C')
        
        # scale information
        pixel_size_microns = self.telescope.camera.pixel_size_m * 1e6
        header['XPIXSZ'] = (pixel_size_microns, 'Pixel Width in microns (after binning)')
        header['YPIXSZ'] = (pixel_size_microns, 'Pixel Height in microns (after binning)')

        # Binning
        header['XBINNING'] = (1, 'Binning factor in width')
        header['YBINNING'] = (1, 'Binning factor in height')
        header['XORGSUBF'] = (0, 'Subframe X position in binned pixels')
        header['YORGSUBF'] = (0, 'Subframe Y position in binned pixels')

        # readout mode
        header['READOUTM'] = ('Normal', 'Readout mode of image')

        # filter information
        header['FILTER'] = (self.photon.band_name, 'Filter used when taking image')

        # image type
        is_calibration = r['target_name'] in ['bias', 'flat', 'dark']
        if is_calibration:
            if r['target_name'] == 'bias':
                image_type = 'Bias Frame'
            elif r['target_name'] == 'dark':
                image_type = 'Dark Frame'
            elif r['target_name'] == 'flat':
                image_type = 'Flat Frame'
        else:
            image_type = 'Light Frame'
        header['IMAGETYP'] = (image_type, 'Type of image')

        # telescope optics information
        focal_length_mm = self.telescope.focal_length_m * 1000
        aperture_diameter_mm = self.telescope.diameter_m * 1000
        aperture_area_mm2 = np.pi * (aperture_diameter_mm / 2) ** 2

        header['FOCALLEN'] = (focal_length_mm, 'Focal length of telescope in mm')
        header['APTDIA'] = (aperture_diameter_mm, 'Aperture diameter of telescope in mm')
        header['APTAREA'] = (aperture_area_mm2, 'Aperture area of telescope in mm^2')

        # camera settings
        header['EGAIN'] = (self.telescope.camera.gain, 'Electronic gain in e-/ADU')
        header['GAIN'] = (0, 'Gain setting (camera-specific)')  
        header['OFFSET'] = (int(self.telescope.mount.goto_error_arcsec/(self.telescope.camera.pixel_size_m/self.telescope.focal_length_m*206265)), 'Offset setting')

        # software information
        header['SBSTDVER'] = ('SBFITSEXT Version 1.0', 'Version of SBFITSEXT standard in effect')
        header['SWCREATE'] = ('Tianyu Image Simulator v1.0', 'Name of software')
        header['SWSERIAL'] = ('SIMULATOR-2024-TIANYU', 'Software serial number')

        # Focuser information (if any)
        header['FOCUSSSZ'] = (10.0, 'Focuser step size in microns')

        # Observatory location information
        header['SITELAT'] = (self.telescope.latlonalt[0], 'Latitude of the imaging location')
        header['SITELONG'] = (self.telescope.latlonalt[1], 'Longitude of the imaging location')

        header['JD'] = (self.time, 'Julian Date at time of exposure')
        header['OBJECT'] = (r['target_name'], 'Object name')
        header['TELESCOP'] = (self.telescope.telescope_name, 'telescope used to acquire this image')
        header['INSTRUME'] = (self.telescope.camera.camera_name, 'Instrument/camera name')

        # Observer information
        header['OBSERVER'] = ('simulator', 'Observer name')
        header['NOTES'] = ('Simulated image', 'Additional notes')
        header['SWOWNER'] = ('Tianyu Telescope Team', 'Licensed owner of software')


        # telescope pointing information (only for science images)
        if not is_calibration:
            header['RA'] = (self.telescope.mount.ra_deg, 'Right Ascension (deg)')
            header['DEC'] = (self.telescope.mount.dec_deg, 'Declination (deg)')

            # # Altitude and Azimuth
            # try:
            #     altaz_frame = AltAz(obstime=time_obj, location=self.telescope.position)
            #     pointing = SkyCoord(ra=self.telescope.mount.ra_deg*u.deg, 
            #                       dec=self.telescope.mount.dec_deg*u.deg,
            #                       frame='icrs', obstime=time_obj, 
            #                       location=self.telescope.position)
            #     altaz = pointing.transform_to(altaz_frame)

            #     header['OBJCTALT'] = (altaz.alt.deg, 'Object altitude (deg)')
            #     header['OBJCTAZ'] = (altaz.az.deg, 'Object azimuth (deg)')
            #     header['AIRMASS'] = (altaz.secz.value, 'Airmass at observation')
            # except:
            #     pass

        header['RDNOISE'] = (self.telescope.camera.read_noise_e, 'Read noise (e-)')
        header['DARKCUR'] = (self.telescope.camera.dark_current_e_s_1, 'Dark current (e-/s/pixel)')
        header['MAXADU'] = (2**self.telescope.camera.bit_per_pixel - 1, 'Maximum ADU value')
        header['READTIME'] = (self.telescope.camera.readout_time_s, 'Readout time (s)')

        # WCS information
        if not is_calibration:
            try:
                wcs_obj = self.telescope.wcs
                header['CTYPE1'] = ('RA---TAN', 'WCS projection type for axis 1')
                header['CTYPE2'] = ('DEC--TAN', 'WCS projection type for axis 2')
                header['CRPIX1'] = (wcs_obj.wcs.crpix[0], 'Reference pixel on axis 1')
                header['CRPIX2'] = (wcs_obj.wcs.crpix[1], 'Reference pixel on axis 2')
                header['CRVAL1'] = (wcs_obj.wcs.crval[0], 'Coordinate value at reference pixel (deg)')
                header['CRVAL2'] = (wcs_obj.wcs.crval[1], 'Coordinate value at reference pixel (deg)')
                header['CDELT1'] = (wcs_obj.wcs.cdelt[0], 'Coordinate increment along axis 1 (deg)')
                header['CDELT2'] = (wcs_obj.wcs.cdelt[1], 'Coordinate increment along axis 2 (deg)')

                if hasattr(wcs_obj.wcs, 'pc'):
                    header['PC1_1'] = (wcs_obj.wcs.pc[0, 0], 'Linear transformation matrix element')
                    header['PC1_2'] = (wcs_obj.wcs.pc[0, 1], 'Linear transformation matrix element')
                    header['PC2_1'] = (wcs_obj.wcs.pc[1, 0], 'Linear transformation matrix element')
                    header['PC2_2'] = (wcs_obj.wcs.pc[1, 1], 'Linear transformation matrix element')

                header['WCSAXES'] = (2, 'Number of WCS axes')
                header['RADESYS'] = ('ICRS', 'Coordinate system')
            except:
                pass

        return header

    def load_event(self,fp,type):
        pass

    def run_sim(self):
        def wait_to_ok(device):
            epsilon = 1e-9
            if device.status == 'busy':
                return device.idle_time+epsilon
            else:
                return self._time
        if not 'simulator' in self.name:
            raise   RuntimeError('Simulator only works in simulator world')
        ct_name = {}
        for i,r in self.schedule.iterrows():
            if self.time > r['jd_utc_end']:
                continue
            else:
                if self.time < r['jd_utc_begin']: # wait until the schedule starts
                    self._time = r['jd_utc_begin']

                if r['target_name']!='bias' and r['target_name']!='flat':
                    if r['ra']==r['ra'] and r['dec']==r['dec']:
                        ra = r['ra']
                        dec = r['dec']
                        
                    if r['alt']==r['alt'] and r['azi']==r['azi']:
                        alt = r['alt']
                        azi = r['azi']
                        coord = SkyCoord(alt=alt*u.deg, az=azi*u.deg, obstime=Time(self._time,scale='utc',format='jd'), frame='altaz', location=self.telescope.position)
                        ra_dec = coord.transform_to('icrs')
                        ra = ra_dec.ra.deg
                        dec = ra_dec.dec.deg

                    altaz_frame = AltAz(obstime=self.time_astropy, location=self.telescope.position)
                    alt = SkyCoord(ra=ra * u.deg, dec=dec * u.deg,
                                            frame='icrs', obstime=self.time_astropy, location=self.telescope.position).transform_to(altaz_frame).alt.degree 
                    if alt < 25:
                        raise ValueError(f"Target {r.get('target_name','<unknown>')} at time {self.time} has altitude {alt}° (<25°). Aborting observation.")
                    
                    self.telescope.mount.goto(ra, dec)
                self._time = wait_to_ok(self.telescope.mount)
                ct_name[r['target_name']] = 0
                total_frames = r['n_max_frames'] if r['n_max_frames'] > 0 else None
                with tqdm(total=total_frames, desc=r['target_name']) as pbar:
                    while self.time < r['jd_utc_end'] and (r['n_max_frames']<=0 or ct_name[r['target_name']]< r['n_max_frames']):
                        start_timestamp = sys_time.time()
                        ct_name[r['target_name']] += 1
                        target_dir = Path(self.output) / r['target_name']
                        target_dir.mkdir(parents=True, exist_ok=True)
                        self.target_dir = target_dir
                        self.schedule_target_name = r['target_name']
                        out = target_dir
                        self.telescope.camera.capture(r['exposure_s'])
                        self._time = wait_to_ok(self.telescope.camera)
                        image = self.telescope.camera.download(self.img_simulator.simulate_full_chain(frame_type=r['target_name'], flat_level=r['flat_level']))
                        header = self.create_fits_header(r)
                        fits.writeto(out / (r['target_name'] + '-' + str(ct_name[r['target_name']]) + '.fits'),  image.astype(np.int32), overwrite=True, header=header)
                        print(r['target_name'],'captured at',self.time)
                        self._time = self._time + r['delay_between_frame_s']/24/3600

                        end_timestamp = sys_time.time()
                        duration = end_timestamp - start_timestamp
                        log_file = Path(out) / "capture_timing_log.csv"
                        with open(log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                r['target_name'], 
                                ct_name[r['target_name']], 
                                f"{duration:.4f}"
                            ])

                        pbar.update(1) 

                # preprocessing image (cut pixel, stacking)
                # push it into Rabbitmq
                
                # TBD
                ################################################################



    def run_real(self):
        if 'simulator' in self.name:
            raise   RuntimeError('Real world only works in real world')
        

class photons_distribution_simulator:
    gaia_dict = {}
    vsx_dict = {}
    galaxy_sn_dict = {}
    def __init__(self, photons_config):
        self.initpar = photons_config
        for key, val in photons_config.items():
            setattr(self, key, val) 
        df = pd.read_csv(self.filter)
        wavelength = np.asarray(df['wavelength']) 
        transmission = np.asarray(df['throughput_frac'])
        nu = 3e8 / (wavelength * 1e-9)  # Hz
        idx = np.argsort(nu)
        nu_sorted = nu[idx]
        T_sorted = transmission[idx]
        T_lamda = np.trapz(T_sorted, nu_sorted)
        self.T_lambda = T_lamda
        
    def connect(self, world):
        self.world = world
        self.hnu = 4.45765*6.62607015e-20
        self.target_key = (self.world.telescope.mount.ra_deg, self.world.telescope.mount.dec_deg)
        self.FWHM = self.world.telescope.seeing_arcsec / self.world.telescope.arcsec_pixel_1
        # self.FWHM_origin = self.world.telescope.seeing_arcsec / self.world.telescope.arcsec_pixel_1
        # self.FWHM = self.FWHM_origin + np.random.normal(0, 0.15*self.FWHM_origin)
        # if self.FWHM < (0.7*self.FWHM_origin) or self.FWHM > (1.3*self.FWHM_origin):
        #     self.FWHM = self.FWHM_origin
        # alpha_origin = self.alpha
        # self.alpha = self.alpha_origin + np.random.normal(0, 0.1*self.alpha_origin)
        # if self.alpha < (0.7*alpha_origin) or self.alpha > (1.3*alpha_origin):
        #     self.alpha = alpha_origin
        self.gamma = self.FWHM / (2 * np.sqrt(2**(1/self.alpha) - 1))
    
    @classmethod
    def update_gaia_data_dict(cls, target_key, data):
        cls.gaia_dict['target_key'] = target_key
        cls.gaia_dict['gaia_data'] = data

    @classmethod
    def update_vsx_data_dict(cls, target_key, star, id, parameters):
        cls.vsx_dict['target_key'] = target_key
        cls.vsx_dict['star'] = star
        cls.vsx_dict['id'] = id
        cls.vsx_dict['parameters'] = parameters

    @classmethod
    def update_galaxy_sn_dict(cls, target_key, select_data):
        cls.galaxy_sn_dict['target_key'] = target_key
        cls.galaxy_sn_dict['select_data'] = select_data

    def sky(self,img):# to simplify the problem, we set the image to align with dec axis
        def get_direction(ra1,dec1,ra2,dec2):
            ra1_rad = ra1 * np.pi / 180
            dec1_rad = dec1 * np.pi / 180
            ra2_rad = ra2 * np.pi / 180
            dec2_rad = dec2 * np.pi / 180
            eRA = np.array([-np.sin(ra1_rad),np.cos(ra1_rad),0]).reshape(-1,1)
            eDEC = np.array([-np.sin(dec1_rad)*np.cos(ra1_rad),-np.sin(dec1_rad)*np.sin(ra1_rad),np.cos(dec1_rad)]).reshape(-1,1)
            r1 = np.array([np.cos(dec1_rad)*np.cos(ra1_rad),np.cos(dec1_rad)*np.sin(ra1_rad),np.sin(dec1_rad)]).reshape(-1,1)
            r2 = np.array([np.cos(dec2_rad)*np.cos(ra2_rad),np.cos(dec2_rad)*np.sin(ra2_rad),np.sin(dec2_rad)]).reshape(-1,1)
            direction = r2 - r1
            comp_ra = np.squeeze(eRA.T.dot(direction))
            comp_dec = np.squeeze(eDEC.T.dot(direction))
            norm = np.sqrt(comp_ra**2+comp_dec**2)
            comp_ra,comp_dec = comp_ra/norm, comp_dec/norm
            return comp_ra,comp_dec
        #consider sun and moon
        def moon_sky(rho_moon_deg,Z,Zm,alpha): # https://arxiv.org/pdf/1304.7107
            def f(rho):
                PA = 1.5
                PB = 0.9
                rho_rad = rho/180*np.pi
                fR = 10**5.36*(1.06+np.cos(rho_rad)**2)
                if rho>=10:
                    fM = 10**(6.15-rho/40)
                else:
                    fM = 6.2*10**7*rho**(-2)
                return PA*fM+PB*fR
                
            def X(Z):
                Z_rad = Z*np.pi/180
                return (1-0.96*np.sin(Z_rad)**2)**(-0.5)

            K = self.extinction_coeffcient
            m = -12.73+0.026*np.abs(alpha)+4*10**(-9)*alpha**4
            I_star = 10**(-0.4*(m+16.57))
            B_moon = f(rho_moon_deg)*I_star*10**(-0.4*K*X(Zm))*(1-10**(-0.4*K*X(Z)))
            V_moon = (20.7233-np.log(B_moon/34.08))/0.92104
            return V_moon
       
        # get moon phase
        sun = get_body('sun',self.world.time_astropy)#.transform_to(Altaz_obs)
        moon = get_body('moon',self.world.time_astropy)#.transform_to(Altaz_obs)
        # Compute elongation (angular separation between Sun and Moon)
        elongation = sun.transform_to(self.world.telescope.Altaz_obs).separation(moon.transform_to(self.world.telescope.Altaz_obs)).deg  
        alpha = 180 - elongation
        rho = self.world.telescope.pointing.separation(moon).deg
        Zm = 90-moon.transform_to(self.world.telescope.Altaz_obs).alt.deg
        Z = 90-self.world.telescope.pointing.transform_to(self.world.telescope.Altaz_obs).alt.deg

        # Compute the Sun
        sun_altaz = sun.transform_to(self.world.telescope.Altaz_obs)
        sun_alt = sun_altaz.alt.deg
        sun_az = sun_altaz.az.deg
        obs_az = self.world.telescope.pointing.transform_to(self.world.telescope.Altaz_obs).az.deg
        theta = (sun_az-obs_az)
        if theta<0:
            theta += 360
        sep_sun_obs_az = np.min(((360-theta),theta))#https://arxiv.org/pdf/1407.8283 
        G_Sun = -2.5/np.log(10)*10**(-0.005555*sep_sun_obs_az-1)/3600     # mag per arcsec
        M_sun = np.min([30,np.max([1,8-1.03*sun_alt])])#https://arxiv.org/pdf/1407.8283
        racomp, deccomp = get_direction(self.world.telescope.pointing.ra.deg,self.world.telescope.pointing.dec.deg,sun.ra.deg,sun.dec.deg)
        dSun_dRA = G_Sun*racomp
        dSun_dDEC = G_Sun*deccomp
        # print(f'sep to moon = {rho} deg; \nphase of Moon = {alpha}\nZenith distance of moon = {Zm}\nAzi of moon = {moon.transform_to(Altaz_obs).az.deg}\nAlt of moon = {moon.transform_to(Altaz_obs).alt.deg}')
        # print(f'pointing alt = {90-Z}; pointing az = {obs_az}')
        # print(f"Telescope RA: {coord_pointing.ra.deg}, DEC: {coord_pointing.dec.deg}")
        # print(f"Moon RA: {moon.ra.deg}, DEC: {moon.dec.deg}")
        # print(M_sun,sun_alt)
        # print(dMoon_dRA,dMoon_dDEC,dSun_dRA,dSun_dDEC)

        ny, nx = img.shape
        mode = self.sky_background_mode

        if mode == 'precise':
            M_moon_field = np.zeros((ny, nx))

            # Calculate moon sky brightness for each pixel using for loop
            center_ra = self.world.telescope.mount.ra_deg
            center_dec = self.world.telescope.mount.dec_deg

            for i in range(ny):
                for j in range(nx):
                    # Calculate RA, DEC for this pixel
                    delta_ra_arcsec = (j - nx//2) * self.world.telescope.arcsec_pixel_1
                    delta_dec_arcsec = (i - ny//2) * self.world.telescope.arcsec_pixel_1

                    # Convert to degrees and account for declination scaling
                    delta_ra_deg = delta_ra_arcsec / 3600.0 / np.cos(np.radians(center_dec))
                    delta_dec_deg = delta_dec_arcsec / 3600.0

                    pixel_ra = center_ra + delta_ra_deg
                    pixel_dec = center_dec + delta_dec_deg

                    # Create SkyCoord for this pixel
                    pixel_coord = SkyCoord(ra=pixel_ra*u.deg, dec=pixel_dec*u.deg,
                                         obstime=self.world.time_astropy, frame='icrs',
                                         location=self.world.telescope.position)

                    # Calculate separation from moon for this pixel
                    rho_pixel = pixel_coord.separation(moon).deg

                    # Calculate zenith distance for this pixel
                    pixel_altaz = pixel_coord.transform_to(self.world.telescope.Altaz_obs)
                    Z_pixel = 90 - pixel_altaz.alt.deg

                    # Calculate moon sky brightness for this pixel
                    M_moon_field[i, j] = moon_sky(rho_pixel, Z_pixel, Zm, alpha)

        if mode == 'fast':
            M_moon = moon_sky(rho,Z,Zm,alpha)
            coord_pointing_ra_deviated = SkyCoord(ra=(self.world.telescope.mount.ra_deg+1/3600)*u.deg, dec=self.world.telescope.mount.dec_deg*u.deg, obstime=self.world.    time_astropy, frame='icrs', location=self.world.telescope.position)
            coord_pointing_dec_deviated = SkyCoord(ra=self.world.telescope.mount.ra_deg*u.deg, dec=(self.world.telescope.mount.dec_deg+1/3600)*u.deg, obstime =self.world.  time_astropy, frame='icrs', location=self.world.telescope.position)
            rho_ra_deviated = coord_pointing_ra_deviated.separation(moon).deg
            rho_dec_deviated = coord_pointing_dec_deviated.separation(moon).deg
            Z_ra_deviated = 90-coord_pointing_ra_deviated.transform_to(self.world.telescope.Altaz_obs).alt.deg
            Z_dec_deviated = 90-coord_pointing_dec_deviated.transform_to(self.world.telescope.Altaz_obs).alt.deg
            M_moon_ra_deviated = moon_sky(rho_ra_deviated,Z_ra_deviated,Zm,alpha)    
            M_moon_dec_deviated = moon_sky(rho_dec_deviated,Z_dec_deviated,Zm,alpha)
            dMoon_dRA = (M_moon_ra_deviated-M_moon) #mag per arcsec
            dMoon_dDEC = (M_moon_dec_deviated-M_moon) #mag per arcsec

            # Generate pixel grid
            x = np.arange(nx)
            y = np.arange(ny)
            xx, yy = np.meshgrid(x, y)
            arcsec_pixel_1 = self.world.telescope.arcsec_pixel_1
            M_moon_field = M_moon + dMoon_dRA*(xx-nx//2)*arcsec_pixel_1 + dMoon_dDEC*(yy-ny//2)*arcsec_pixel_1  ## mag

        M_raw_sky = self.sky_raw_mag
        M_sun_field = M_sun + dSun_dRA*(xx-nx//2)*arcsec_pixel_1 + dSun_dDEC*(yy-ny//2)*arcsec_pixel_1
        M_all = -2.5*np.log10(10**(-0.4*M_raw_sky)+10**(-0.4*M_moon_field))
        +10**(-0.4*M_sun_field)
        print(f'Sky brightness mag/arcsec^2: {M_all}')
        # flux_c = (3e8/self.delta_lambda_min - 3e8/self.delta_lambda_max) * 1e-3 * 10**(-0.4*(M_all-self.zero_mag))
        flux_c = self.T_lambda * 1e-3 * 10**(-0.4*(M_all-self.zero_mag))
        
        print(f'Sky brightness flux: {flux_c}')
        c = flux_c * np.pi * (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s / self.hnu
        print(f'Sky brightness photons: {c}')
        if False:
            import matplotlib.pyplot as plt
            plt.imshow(M_all)
            plt.colorbar()
            plt.show()
        return c
    
    def scintillation(self, img):
        CY = self.CY
        scintillation_L0 = self.scintillation_L0
        scintillation_method = self.scintillation_method
        scintillation_seed = self.scintillation_seed

        alt = self.world.telescope.pointing.transform_to(self.world.telescope.Altaz_obs).alt.rad
        z = np.pi/2 - alt
        z = np.clip(z, 0, np.pi/2)  # limit in [0°,90°]

        cos_z = np.cos(z)
        cos_z = np.clip(cos_z, 0.05, 1.0)  # Avoiding numerical problems when approaching zero

        diameter = self.world.telescope.diameter_m
        exposure = self.world.telescope.camera.exposure_s
        height = self.world.telescope.position.height.to(u.m).value

        # Safe calculation of exponent
        exponent = -2 * height / 8000
        exponent = np.clip(exponent, -100, 100)  # Prevent excessive exponent

        # Safe calculation of scintillation
        term1 = 10**(-6) * CY**2
        term2 = diameter**(-4/3)
        term3 = 1 / exposure
        term4 = cos_z**(-3)
        term5 = np.exp(exponent)
        print(f'alt: {alt}, z: {z}, term1: {term1}, term2: {term2}, term3: {term3}, term4: {term4}, term5: {term5}')

        # Make sure the square root parameter is positive
        scintillation_value = term1 * term2 * term3 * term4 * term5
        if scintillation_value < 0:
            print('scintillation_value < 0: error happen, check the pointing is below the horizon.')
            return np.ones(img.shape)

        sigma_I = np.sqrt(scintillation_value)
        # sigma_I = 0

        def _kolmogorov_amplitude_filter(nx, ny, L0):
            """
            Construct Kolmogorov amplitude filter A(k) = (k² + k₀²)^(-11/12)

            Parameters:
            -----------
            nx, ny : int
                Grid dimensions
            L0 : float
                Outer scale in pixels

            Returns:
            --------
            A_k : ndarray (complex)
                Amplitude filter in frequency domain
            """
            # Generate frequency grids (FFT convention)
            kx = fft.fftfreq(nx, d=1) * 2 * np.pi  # cycles per pixel -> radians per pixel: f=2*np.pi*omega
            ky = fft.fftfreq(ny, d=1) * 2 * np.pi

            KX, KY = np.meshgrid(kx, ky)

            # Radial frequency
            k = np.sqrt(KX**2 + KY**2)

            # Outer scale cutoff frequency
            k0 = 2 * np.pi / L0

            # Kolmogorov amplitude spectrum: A(k) ∝ (k² + k₀²)^(-11/12)
            # Add small epsilon to avoid division by zero at DC
            epsilon = 1e-10
            A_k = (k**2 + k0**2 + epsilon)**(-11/12)

            # # Handle DC component (k=0) separately to avoid issues
            # A_k[0, 0] = A_k[0, 1]  # Use nearby value

            return A_k

        def _lognormal_mapping(H, sigma_I):
            """
            Map normalized field H to lognormal multiplicative field
            This ensures f(x,y) > 0 always and provides exact control over
            mean and variance: E[f] = 1, Var(f) = sigma_I²
            """

            # Lognormal parameters
            a = np.sqrt(np.log(1 + sigma_I**2))

            # Lognormal mapping: f = exp(a*H - a²/2)
            # This gives E[f] = 1 and Var(f) = sigma_I²
            f = np.exp(a * H - a**2 / 2)

            return f

        def _linear_mapping(H, sigma_I):
            """
            Linear mapping: f = 1 + sigma_I * H
            """

            f = 1.0 + sigma_I * H
            # average = 1
            # Var(f) = Var(1 + sigma_I * H) = sigma_I^2 * Var(H) = sigma_I^2

            # Warn if negative values occur
            if np.any(f <= 0):
                n_negative = np.sum(f <= 0)
                print(f"Warning: {n_negative} pixels have non-positive values in linear mapping")
                # Clip to small positive value
                f = np.maximum(f, 0.01) # Replace all values in f that are less than 0.01 with 0.01.

            return f

        # in most condition ,the seed should be None: each frame scintillation should be independent.
        if scintillation_seed is not None:
            np.random.seed(scintillation_seed)

        ny, nx = img.shape
        # Set default outer scale if not provided, the typical vale: few meters to kilometers
        if scintillation_L0 is None:
            scintillation_L0 = 100 / self.world.telescope.camera.pixel_size_m
        print(f"Generating scintillation field: {nx}×{ny}, σ_I={sigma_I:.4f}, L₀={scintillation_L0:.1f} pixels")
        # Step 1: Generate real-valued white noise field
        w = np.random.normal(0, 1, size=(ny, nx))
        # Step 2: FFT to frequency domain
        W_tilde = fft.fft2(w)  # fft 2D
        # Step 3: Construct Kolmogorov amplitude filter
        A_k = _kolmogorov_amplitude_filter(nx, ny, scintillation_L0)
        # Step 4: Apply filter and IFFT back to spatial domain
        H_filtered = W_tilde * A_k
        h = np.real(fft.ifft2(H_filtered))
        # Step 5: Normalize the field (zero mean, unit variance)
        h_mean = np.mean(h)
        h_std = np.std(h)
        if h_std == 0:
            # Degenerate case - return uniform field
            return np.ones(img.shape)
        H_normalized = (h - h_mean) / h_std
        # Step 6: Map to positive multiplicative field
        if scintillation_method == 'lognormal':
            f = _lognormal_mapping(H_normalized, sigma_I)
        elif scintillation_method == 'linear':
            f = _linear_mapping(H_normalized, sigma_I)
        else:
            raise ValueError(f"Unknown method: {scintillation_method}. Use 'lognormal' or 'linear'")
        
        if np.any(np.isnan(f)):
            print("Warning: scintillation field contains NaN. Replacing with 1.0.")
            print(f'the total number of NaN values: {np.sum(np.isnan(f))}')
            f = np.nan_to_num(f, nan=1.0)
        if np.any(f <= 0):
            print("Warning: scintillation field contains non-positive values. Clipping to 1-sigma_I.")
            print(f"the total number of non-positive values: {np.sum(f <= 0)}")
            f = np.clip(f, 1-sigma_I, None)
        # Verify statistics
        actual_mean = np.mean(f)
        actual_var = np.var(f)
        actual_sigma = np.sqrt(actual_var)
        print(f"check scintillation field statistics: mean={actual_mean:.4f}, σ={actual_sigma:.4f}")
        return f
        
    def jitter_error(self, r):
        total_electron_gaia = 4.42 * r['phot_g_n_obs'] * r['phot_g_mean_flux']
        error_photometric = np.sqrt(total_electron_gaia)/r['phot_g_n_obs']
        error_total = r['phot_g_mean_flux_error']
        error_jitter = np.sqrt(np.maximum(0,error_total**2-error_photometric**2))
        error_fraction_jitter = error_jitter/r['phot_g_mean_flux'] * (~(r['phot_variable_flag']=='VARIABLE')) # Variable star have larger error, process correspondingly 
        error_fraction_jitter_sample = 1+error_fraction_jitter * np.random.randn(len(r))
        return error_fraction_jitter_sample

    def ADR(self, t_tdb_jd, ra, dec):
        """
        Atmospheric refraction calculator based on Stone (1996) method.

        This part implements the accurate method for computing atmospheric refraction
        as described in "An Accurate Method for Computing Atmospheric Refraction"
        by Ronald C. Stone (PASP, 1996).
        """

        def water_vapor_pressure_from_dewpoint(dewpoint_c):
            """
            Calculate water vapor pressure from dew point temperature.

            Args:
                dewpoint_c (float): Dew point in Celsius

            Returns:
                float: Water vapor pressure in mmHg
            """
            td = dewpoint_c
            pw = (4.50874 + 0.341724*td + 0.0106778*td**2 + 
                  0.000184889*td**3 + 0.00000238294*td**4 + 
                  0.0000000203447*td**5)
            return pw

        def dewpoint_from_humidity(temp_c, relative_humidity):
            """
            Calculate dew point from temperature and relative humidity.

            Args:
                temp_c (float): Temperature in Celsius
                relative_humidity (float): Relative humidity in percent (0-100)

            Returns:
                float: Dew point in Celsius
            """
            t = temp_c
            rh = relative_humidity
            x = np.log(rh / 100.0)

            td = 238.3 * ((t + 238.3) * x + 17.2694 * t) / \
                 ((t + 238.3) * (17.2694 - x) - 17.2694 * t)

            return td

        def refractive_index(wavelength_angstrom, temp_c, pressure_mm, 
                            water_vapor_pressure_mm):
            """
            Calculate refractive index of air using Owens (1967) formulation.

            Args:
                wavelength_angstrom (float): Wavelength in Angstroms
                temp_c (float): Temperature in Celsius
                pressure_mm (float): Atmospheric pressure in mmHg
                water_vapor_pressure_mm (float): Water vapor pressure in mmHg

            Returns:
                float: Refractive index of air
            """
            # Convert units
            T = 273.15 + temp_c  # Temperature in Kelvin
            Ps = 1.333224 * (pressure_mm - water_vapor_pressure_mm)  # Dry air pressure in mb
            Pw = 1.333224 * water_vapor_pressure_mm  # Water vapor pressure in mb

            # Wavelength in micrometers
            lambda_um = wavelength_angstrom / 10000.0
            sigma = 1 / lambda_um  # Wave number in reciprocal micrometers
            # Density terms
            Ds = (Ps / T) * (1 + Ps * (57.90e-8 - 9.3250e-4 / T + 0.25844 / T**2))

            Dw = (Pw / T) * (1 + Pw * (1 + 3.7e-4 * Pw) * 
                            (-2.37321e-3 + 2.23366 / T - 710.792 / T**2 + 7.75141e4 / T**3))

            # Refractive index calculation
            n_minus_1 = 1e-8 * (
                (2371.34 + 683939.7 / (130 - sigma**2) + 4547.3 / (38.9 - sigma**2)) * Ds +
                (6487.31 + 58.058*sigma**2 - 0.71150*sigma**4 + 0.08851*sigma**6) * Dw
            )

            return 1.0 + n_minus_1

        def gravity_correction(latitude_deg, elevation_m):
            """
            Calculate gravity correction factor kappa.

            Args:
                latitude_deg (float): Astronomical latitude in degrees
                elevation_m (float): Elevation above sea level in meters

            Returns:
                float: Gravity correction factor kappa
            """
            phi_rad = np.radians(latitude_deg)
            kappa = (1 + 0.005302 * np.sin(phi_rad)**2 - 
                    0.00000583 * np.sin(2*phi_rad)**2 - 
                    0.000000315 * elevation_m)
            return kappa

        def atmospheric_refraction(jd, ra_deg, dec_deg, wavelength_angstrom,
                                 temp_c, pressure_mm, water_vapor_pressure_mm,
                                 latitude_deg=0.0, longitude_deg=0.0, elevation_m=0.0):
            """
            Calculate atmospheric refraction for a single wavelength.

            Args:
                wavelength_angstrom (float): Wavelength in Angstroms
                temp_c (float): Temperature in Celsius
                pressure_mm (float): Atmospheric pressure in mmHg
                water_vapor_pressure_mm (float): Water vapor pressure in mmHg
                latitude_deg (float): Astronomical latitude in degrees (default: 0)
                elevation_m (float): Elevation above sea level in meters (default: 0)

            Returns:
                Args: Atmospheric refraction in arcseconds
            """

            alt = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, obstime=self.world.time_astropy, frame='icrs', location=self.world.telescope.position).transform_to(self.world.telescope.Altaz_obs).alt.rad
            z0_rad = np.pi/2 - alt

            # if z0_deg >= 75.0:
            #     print("Warning: Refraction calculation may be inaccurate for zenith distances >= 75°")

            tan_z0 = np.tan(z0_rad)

            # Calculate refractive index
            n0 = refractive_index(wavelength_angstrom, temp_c, pressure_mm, 
                                      water_vapor_pressure_mm)

            # Calculate beta parameter
            beta = 0.001254 * (273.15 + temp_c) / 273.15

            # Calculate gravity correction
            kappa =gravity_correction(latitude_deg, elevation_m)


            # Calculate refraction using first two terms of expansion
            R_lambda = ((n0 - 1) * kappa * (1 - beta) * tan_z0 - 
                       (n0 - 1) * kappa * (beta - (n0-1)/2) * tan_z0**3)

            # Convert from radians to arcseconds
            R_arcsec = R_lambda * 206264.8

            return R_arcsec


        def equatorial_refraction_components(ra_deg, dec_deg, jd, 
                                           latitude_deg, longitude_deg, total_refraction_arcsec):
            """
            Calculate refraction components in right ascension and declination.

            Args:
                ra_deg (float): Right ascension in degrees
                dec_deg (float): Declination in degrees
                lst_deg (float): Local sidereal time in degrees
                latitude_deg (float): Observatory latitude in degrees
                total_refraction_arcsec (float): Total refraction in arcseconds

            Returns:
                tuple: (delta_ra_arcsec, delta_dec_arcsec) - Refraction corrections
            """

            t = Time(jd, format='jd', scale='utc')

            # 计算地方恒星时（apparent 或 mean）
            # 'apparent' 使用春分章动/章动修正，更适合精确天球坐标变换
            lst_deg = t.sidereal_time('apparent', longitude=longitude_deg * u.deg)

            # Convert to radians
            ra_rad = np.radians(ra_deg)
            dec_rad = np.radians(dec_deg)
            lst_rad = np.radians(lst_deg)
            phi_rad = np.radians(latitude_deg)

            # Calculate hour angle
            ha_rad = lst_rad.to(u.rad).value - ra_rad

            # Calculate zenith distance
            z0_rad = np.arccos(np.sin(dec_rad) * np.sin(phi_rad) + np.cos(dec_rad) * np.cos(phi_rad) * np.cos(ha_rad))
            cos_z0 = np.cos(z0_rad)
            sin_z0 = np.sin(z0_rad)

            # Calculate parallactic angle components
            sin_psi = np.cos(phi_rad) * np.sin(ha_rad) / sin_z0
            cos_psi = (np.sin(phi_rad) - np.sin(dec_rad) * cos_z0) / (np.cos(dec_rad) * sin_z0)

            # Convert refraction to radians
            R_rad = np.radians(total_refraction_arcsec / 3600)

            # Calculate refraction components
            delta_ra_rad = -R_rad * sin_psi / np.cos(dec_rad)
            delta_dec_rad = -R_rad * cos_psi

            # Convert back to arcseconds
            delta_ra_arcsec = delta_ra_rad * 206264.8 / 3600
            delta_dec_arcsec = delta_dec_rad * 206264.8 / 3600

            return delta_ra_arcsec, delta_dec_arcsec


        # Standard observing conditions
        temp_c = self.temp_C
        pressure_mm = self.pressure_mm  # mmHg (sea level)
        humidity = self.relative_humidity  # relative humidity
        latitude_deg = self.world.telescope.latlonalt[0]
        longitude_deg = self.world.telescope.latlonalt[1]
        elevation_m = self.world.telescope.latlonalt[2]

        csv_path = self.filter
        band = self.band_name
        df = pd.read_csv(csv_path)
        sub = df[(df.band==band)&(df.throughput_frac>0)]
        lam_set = sub.wavelength.values; wts = sub.throughput_frac.values; wts/=wts.sum()
        lam_eff = np.sum(lam_set*wts)

        # Calculate water vapor pressure from humidity
        dewpoint = dewpoint_from_humidity(temp_c, humidity)
        water_vapor_mm = water_vapor_pressure_from_dewpoint(dewpoint)
        kappa = gravity_correction(latitude_deg, elevation_m)
        n0 = refractive_index(lam_eff, temp_c, pressure_mm, water_vapor_mm)

        print(f"Observing Conditions:")
        print(f"Temperature: {temp_c}°C")
        print(f"Pressure: {pressure_mm} millibars")
        print(f"Relative Humidity: {humidity}%")
        print(f"Dew Point: {dewpoint:.1f}°C")
        print(f"Water Vapor Pressure: {water_vapor_mm:.1f} mmHg")
        print(f"Latitude: {latitude_deg}°")
        print(f"Longitude: {longitude_deg}°")
        print(f"Elevation: {elevation_m} m")
        print(f"Gravity Correction (Kappa): {kappa:.6f}")
        print(f"Index of Refraction: {n0:.6f}")


        total_refraction_arcsec = atmospheric_refraction(
                t_tdb_jd, ra, dec, lam_eff, temp_c, pressure_mm, water_vapor_mm, 
                latitude_deg, longitude_deg, elevation_m
            )
        print(f"ADR: {total_refraction_arcsec}")

        delta = equatorial_refraction_components(ra, dec, t_tdb_jd, latitude_deg, longitude_deg, total_refraction_arcsec)
        print(f"delta_ra_dec: {delta}")
        
        return delta

    def extinction_error(self, ra, dec):
        star_all = SkyCoord(ra=ra, dec=dec, unit='deg', obstime=self.world.time_astropy, frame='icrs', location=self.world.telescope.position)
        z_all = np.pi/2-star_all.transform_to(self.world.telescope.Altaz_obs).alt.rad
        z_all = np.clip(z_all, 0, np.pi/2) # limit in [0°,90°]
        airmass_all = (1.002432*np.cos(z_all)**2+0.148386*np.cos(z_all)+0.0096467)/(np.cos(z_all)**3+0.149864*np.cos(z_all)**2+0.0102963*np.cos(z_all)+0.000303978) #https://opg.optica.org/ao/abstract.cfm?uri=ao-33-6-1108
        K = self.extinction_coeffcient
        luminosity_fraction_extinction = 10**(-0.4*(K*airmass_all))
        print(f'luminosity_fraction_extinction: {luminosity_fraction_extinction}')
        return luminosity_fraction_extinction
    
    # consider flux_variable stars
    def transit(self, t_tdb_jd, transit_catalog, r, r_object_shape):
        transit_relative_flux = np.ones(r_object_shape)
        if type(transit_catalog)==str:
            new_transit_data = []
            transit_events = pd.read_csv(transit_catalog)
            for index,row in transit_events.iterrows():
                if not row['dr3_source_id'] in r['SOURCE_ID']:
                    continue
                
                target_id = row['dr3_source_id']
                target = r[r['SOURCE_ID']==row['dr3_source_id']]
                target_pos = SkyCoord(ra=target['ra'][0]*u.deg, dec=target['dec'][0]*u.deg, frame='icrs',location=self.world.telescope.position)
                ltt_bary = self.world.time_astropy.light_travel_time(target_pos)
                time_barycentre = t_tdb_jd + ltt_bary 
                # Use batman to generate the light curve
                params = batman.TransitParams()
                params.t0 = row['tm_tdb_bjd']                        #time of inferior conjunction
                params.per = row['period_d']                       #orbital period
                params.rp = row['radius_star_radius']                       #planet radius (in units of stellar radii)
                params.a = row['semi_major_axis_stellar_radius']                       #semi-major axis (in units of stellar radii)
                params.inc = row['inclination_deg']                     #orbital inclination (in degrees)
                params.ecc = row['e']                      #eccentricity
                params.w = 90.                        #longitude of periastron (in degrees)
                params.limb_dark = "nonlinear"        #limb darkening model
                params.u = [row['u05'], row['u1'], row['u15'], row['u2']]
                m = batman.TransitModel(params, np.array([time_barycentre.jd]))
                flux = m.light_curve(params)
                transit_relative_flux[np.where(r['SOURCE_ID']==row['dr3_source_id'])] = flux

                new_transit_data.append({
                'target_id': target_id,
                'tdb_jd': t_tdb_jd,
                'tdb_bjd': time_barycentre.jd,
                'normalized_flux': flux
                })

            if new_transit_data:
                csv_transit_path = Path(self.world.target_dir) / 'transit_inputdata_sorted.csv'
                new_df = pd.DataFrame(new_transit_data)
                if csv_transit_path.exists() and csv_transit_path.stat().st_size > 0:
                    try:
                        existing_df = pd.read_csv(csv_transit_path)
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        combined_df = new_df
                else:
                    combined_df = new_df
                combined_df = combined_df.drop_duplicates(subset=['target_id', 'tdb_bjd'], keep='last') ## drop duplicates (based on a combination of target_id and time, keeping the latest flux value)
                combined_df = combined_df.sort_values(['target_id', 'tdb_bjd'])
                combined_df = combined_df.reset_index(drop=True)
                combined_df.to_csv(csv_transit_path, index=False)

        return transit_relative_flux   
 
    def binary(self, t_tdb_jd, binary_catalog, r, r_object_shape):
        binary_relative_flux = np.ones(r_object_shape)
        passband = f'Custom:{self.band_name}'
        if type(binary_catalog)==str:
            df = pd.read_csv(self.filter)
            # band_name = self.band_name
            # wavelength = np.asarray(df['wavelength']) 
            # transmission = np.asarray(df['throughput_frac'])
            # phoebe.create_passband(band_name, 
            #        wavelengths=wavelength, 
            #        transmission=transmission)

            new_binary_data = []
            binary_events = pd.read_csv(binary_catalog)
            #sprint(transit_events)

            for index,row in binary_events.iterrows():
                if not row['dr3_source_id'] in r['SOURCE_ID']:
                    continue

                if row['type'] == 'EA':
                    target_id = row['dr3_source_id']
                    target = r[r['SOURCE_ID']==row['dr3_source_id']]

                    target_pos = SkyCoord(ra=target['ra'][0]*u.deg, dec=target['dec'][0]*u.deg, frame='icrs',location=self.world.telescope.position)
                    ltt_bary = self.world.time_astropy.light_travel_time(target_pos)
                    time_barycentre = t_tdb_jd + ltt_bary

                    # Use phoebe to generate the light curve
                    b = phoebe.default_binary()
                    b.add_dataset('lc', times=[time_barycentre.jd], passband=passband, dataset='lc01')
                    b.set_value('t0_supconj@binary@component', component='binary', value=float(row['t0'])) #  reference time of inferior conjunction
                    b.set_value('period', component='binary', value=float(row['period']))    # orbit period
                    b.set_value('q', component='binary', value=float(row['q']))        # mass ratio
                    b.set_value('incl', component='binary', value=float(row['incl']))   # orbital inclination (in degrees)
                    b.set_value('sma', component='binary', value=float(row['sma']))     # orbital semi-axis

                    b.set_value('teff', component='primary', value=float(row['teff1']))   # primary temperature (K)
                    b.set_value('requiv', component='primary', value=float(row['r1']))    # primary radius (solar radius)
                    b.set_value('teff', component='secondary', value=float(row['teff2']))  # secondary temperature
                    b.set_value('requiv', component='secondary', value=float(row['r2']))  # secondary radius

                    try:
                        b.run_compute()
                    except Exception as err:
                        print(f"Error: {err}")
                    flux = b['value@fluxes@lc01@model'][0]

                    new_binary_data.append({
                    'target_id': target_id,
                    'tdb_jd': t_tdb_jd,
                    'tdb_bjd': time_barycentre.jd,
                    'normalized_flux': flux
                    })

                   
                elif row['type'] == 'EB':
                    target_id = row['dr3_source_id']
                    target = r[r['SOURCE_ID']==row['dr3_source_id']]

                    target_pos = SkyCoord(ra=target['ra'][0]*u.deg, dec=target['dec'][0]*u.deg, frame='icrs',location=self.world.telescope.position)
                    ltt_bary = self.world.time_astropy.light_travel_time(target_pos)

                    time_barycentre = t_tdb_jd + ltt_bary

                    # Use phoebe to generate the light curve
                    for semi_comp in ['primary', 'secondary']:
                        b = phoebe.default_binary()
                        b.add_constraint('semidetached', semi_comp)
                        b.add_dataset('lc', times=[time_barycentre.jd], passband=passband, dataset='lc02')
                       
                        b.set_value('teff', component='primary', value=float(row['teff1'])) 
                        b.set_value('teff', component='secondary', value=float(row['teff2'])) 

                        if semi_comp == 'primary':
                            b.set_value('requiv', component='secondary', value=float(row['r2']))  
                        else:
                            b.set_value('requiv', component='primary', value=float(row['r1']))

                        b.set_value('t0_supconj@binary@component', component='binary', value=float(row['t0']))
                        b.set_value('period', component='binary', value=float(row['period']))
                        b.set_value('q', component='binary', value=float(row['q']))  
                        b.set_value('incl', component='binary', value=float(row['incl']))
                        b.set_value('sma', component='binary', value=float(row['sma']))  
                        try:
                            b.run_compute()
                            success = True
                            break
                        except Exception as e:
                            print(f"semidetached={semi_comp} compute failed: {e}")
                            success = False

                    if not success:
                        # fallback on blackbody atmospheres and manually provide the limb-darkening function and coefficients
                        b.set_value('atm', component='primary', value='blackbody')
                        b.set_value('ld_mode', component='primary', value='manual')
                        b.set_value('ld_func', component='primary', dataset='lc02', value='logarithmic')
                        b.set_value('ld_coeffs', component='primary', dataset='lc02', value=[0.5, 0.5])
                        b.run_compute()

                    flux = b['value@fluxes@lc02@model'][0]

                    new_binary_data.append({
                    'target_id': target_id,
                    'tdb_jd': t_tdb_jd,
                    'tdb_bjd': time_barycentre.jd,
                    'normalized_flux': flux
                    })

                elif row['type'] == 'EW':
                    target_id = row['dr3_source_id']
                    target = r[r['SOURCE_ID']==row['dr3_source_id']]

                    target_pos = SkyCoord(ra=target['ra'][0]*u.deg, dec=target['dec'][0]*u.deg, frame='icrs',location=self.world.telescope.position)
                    ltt_bary = self.world.time_astropy.light_travel_time(target_pos)

                    time_barycentre = t_tdb_jd + ltt_bary

                    # Use phoebe to generate the light curve
                    b = phoebe.default_binary(contact_binary=True)
                    b.add_dataset('lc', times=[time_barycentre.jd], passband=passband, dataset='lc03')
                    # orbit:binary
                    # print(b.filter(component='binary', kind='orbit', context='component'))
                    b.set_value('t0_supconj@binary@component', component='binary', value=float(row['t0']))
                    b.set_value('period', component='binary', value=float(row['period'])) 
                    b.set_value('q',component='binary', value=float(row['q']))  
                    b.set_value('incl', component='binary', value=float(row['incl']))  
                    b.set_value('sma', component='binary', value=float(row['sma']))  

                    # star: primary, secondary
                    # print(b.filter(component='secondary', kind='star', context='component'))
                    b.set_value('teff', component='primary', value=float(row['teff1']))  
                    b.set_value('requiv@primary@component', component='primary', value=float(row['r1']))  
                    b.set_value('teff', component='secondary', value=float(row['teff2']))  

                    try:
                        b.run_compute()
                    except Exception as err:
                        print(f"Error: {err}")
                    flux = b['value@fluxes@lc03@model'][0]
                    
                    new_binary_data.append({
                    'target_id': target_id,
                    'tdb_jd': t_tdb_jd,
                    'tdb_bjd': time_barycentre.jd,
                    'normalized_flux': flux
                    })

                else:
                    print("Unknown type")
            
            if new_binary_data:
                csv_binary_path = Path(self.world.target_dir) / 'binary_inputdata_sorted.csv'
                new_df = pd.DataFrame(new_binary_data)
                if csv_binary_path.exists() and csv_binary_path.stat().st_size > 0:
                    try:
                        existing_df = pd.read_csv(csv_binary_path)
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        combined_df = new_df
                else:
                    combined_df = new_df
                combined_df = combined_df.drop_duplicates(subset=['target_id', 'tdb_jd'], keep='last') ## drop duplicates (based on a combination of target_id and time, keeping the latest flux value)
                combined_df = combined_df.sort_values(['target_id', 'tdb_jd'])
                combined_df = combined_df.reset_index(drop=True)
                combined_df.to_csv(csv_binary_path, index=False)

        return binary_relative_flux
 
    def flare(self, t_tdb_jd, flare_catalog, r, r_object_shape):
        
        # the Llamaradas-Estelares include flare_eqn and flare_model
        def flare_eqn(t,tpeak,fwhm,ampl):
            '''
            The equation that defines the shape for the Continuous Flare Model
            '''
            #Values were fit & calculated using MCMC 256 walkers and 30000 steps

            A,B,C,D1,D2,f1 = [0.9687734504375167,-0.251299705922117,0.22675974948468916,
                              0.15551880775110513,1.2150539528490194,0.12695865022878844]

            # We include the corresponding errors for each parameter from the MCMC analysis

            A_err,B_err,C_err,D1_err,D2_err,f1_err = [0.007941622683556804,0.0004073709715788909,0.0006863488251125649,
                                                      0.0013498012884345656,0.00453458098656645,0.001053149344530907 ]

            f2 = 1-f1

            eqn = ((1 / 2) * np.sqrt(np.pi) * A * C * f1 * np.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2)
                                * special.erfc(((B - t) / C) + (C * D1 / 2))) + ((1 / 2) * np.sqrt(np.pi) * A * C * f2
                                * np.exp(-D2 * t+ ((B / C) + (D2 * C / 2)) ** 2) * special.erfc(((B - t) / C) + (C * D2 / 2)))
            return eqn * ampl


        def flare_model(t,tpeak, fwhm, ampl, upsample=False, uptime=10):
            '''
            The Continuous Flare Model evaluated for single-peak (classical) flare events.
            Use this function for fitting classical flares with most curve_fit
            tools. Reference: Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6

            References
            --------------
            Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
            Davenport et al. (2014) http://arxiv.org/abs/1411.3723
            Jackman et al. (2018) https://arxiv.org/abs/1804.03377

            Parameters
            ----------
            t : 1-d array
                The time array to evaluate the flare over

            tpeak : float
                The center time of the flare peak

            fwhm : float
                The Full Width at Half Maximum, timescale of the flare

            ampl : float
                The amplitude of the flare


            Returns
            -------
            flare : 1-d array
                The flux of the flare model evaluated at each time

                A continuous flare template whose shape is defined by the convolution of a Gaussian and double exponential
                and can be parameterized by three parameters: center time (tpeak), FWHM, and ampitude
            '''

            t_new = (t-tpeak)/fwhm

            if upsample:
                dt = np.nanmedian(np.diff(np.abs(t_new)))
                timeup = np.linspace(min(t_new) - dt, max(t_new) + dt, t_new.size * uptime)

                flareup = flare_eqn(timeup,tpeak,fwhm,ampl)

                # and now downsample back to the original time...

                downbins = np.concatenate((t_new - dt / 2.,[max(t_new) + dt / 2.]))
                flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',bins=np.sort(downbins))
            else:
            
                flare = flare_eqn(t_new,tpeak,fwhm,ampl)

            return flare
        
        flare_relative_flux = np.ones(r_object_shape)
        if type(flare_catalog)==str:
            new_flare_data = []
            flare_events = pd.read_csv(flare_catalog)

            for index,row in flare_events.iterrows():
                if not row['dr3_source_id'] in r['SOURCE_ID']:
                    continue
                target_id = row['dr3_source_id']
                target = r[r['SOURCE_ID']==row['dr3_source_id']]
                
                target_pos = SkyCoord(ra=target['ra'][0]*u.deg, dec=target['dec'][0]*u.deg, frame='icrs',location=self.world.telescope.position)
                ltt_bary = self.world.time_astropy.light_travel_time(target_pos)
                time_barycentre = t_tdb_jd + ltt_bary  

                # Use Llamaradas-Estelares to generate the light curve
                t_peak = row['tpeak']
                fwhm = row['fwhm']
                amplitude = row['amp']
                parameter = [t_peak, fwhm, amplitude]
                flux = flare_model(time_barycentre.jd, *parameter, upsample=False) + 1
                flare_relative_flux[np.where(r['SOURCE_ID']==row['dr3_source_id'])] = flux
                
                new_flare_data.append({
                'target_id': target_id,
                'tdb_jd': t_tdb_jd,
                'tdb_bjd': time_barycentre.jd,
                'normalized_flux': flux
                })

            if new_flare_data:
                csv_flare_path = Path(self.world.target_dir) / 'flare_inputdata_sorted.csv'
                new_df = pd.DataFrame(new_flare_data)
                if csv_flare_path.exists() and csv_flare_path.stat().st_size > 0:
                    try:
                        existing_df = pd.read_csv(csv_flare_path)
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        combined_df = new_df
                else:
                    combined_df = new_df
                combined_df = combined_df.drop_duplicates(subset=['target_id', 'tdb_jd'], keep='last') ## drop duplicates (based on a combination of target_id and time, keeping the latest flux value)
                combined_df = combined_df.sort_values(['target_id', 'tdb_jd'])
                combined_df = combined_df.reset_index(drop=True)
                combined_df.to_csv(csv_flare_path, index=False)

        return flare_relative_flux

    def occultation(self, t_tdb_jd, occultation_catalog, r, r_object_shape):

        def U(n, mu, nu, n_terms=160, tol=1e-13):
            """Lommel U_n 函数"""
            mu, nu = np.broadcast_arrays(np.asarray(mu, float), np.asarray(nu, float))
            q = np.pi * mu * nu
            s = np.zeros_like(q)
            ratio = np.where(nu == 0, 0.0, mu / nu)
            rp = ratio ** n

            for k in range(n_terms):
                term = (-1) ** k * rp * jv(n + 2 * k, q)
                s += term
                if np.all(np.abs(term) < tol):
                    break
                rp *= ratio ** 2
            return s

        Fs_cache={}

        def I_fresnel_diffraction(F, lam, r_pos, r_ast_m):
            """计算 Fresnel 衍射强度"""
            Fs = F(lam); Fs_cache[lam]=Fs
            rho = r_ast_m / Fs
            eta = np.abs(r_pos) / Fs
            beta = 0.5 * np.pi * (eta**2 + rho**2)

            I = np.empty_like(eta)
            inner = eta <= rho

            I[inner] = U(0, eta[inner], rho)**2 + U(1, eta[inner], rho)**2
            U1 = U(1, rho, eta[~inner])
            U2 = U(2, rho, eta[~inner])
            I[~inner] = 1 + U1**2 + U2**2 - 2*U1*np.sin(beta[~inner]) + 2*U2*np.cos(beta[~inner])

            return I

        def overlap_area_circles(R1, R2, d):
            """计算两个圆的重叠面积"""
            d = np.asarray(d)
            A = np.zeros_like(d)
            no = d >= R1 + R2
            full = d <= abs(R1 - R2)
            part = ~(no | full)

            A[full] = np.pi * np.minimum(R1, R2)**2

            if np.any(part):
                dp = d[part]
                t1 = R1**2 * np.arccos((dp**2 + R1**2 - R2**2) / (2 * dp * R1))
                t2 = R2**2 * np.arccos((dp**2 + R2**2 - R1**2) / (2 * dp * R2))
                t3 = 0.5 * np.sqrt((-dp + R1 + R2) * (dp + R1 - R2) * 
                                  (dp - R1 + R2) * (dp + R1 + R2))
                A[part] = t1 + t2 - t3

            return A

         # 主计算逻辑
        occultation_relative_flux = np.ones(r_object_shape)

        if type(occultation_catalog)==str:
            new_occultation_data = []
            occultation_events = pd.read_csv(occultation_catalog)
            for index, row in occultation_events.iterrows():
                if not row['dr3_source_id'] in r['SOURCE_ID']:
                    continue

                # 获取掩星参数
                target_id = row['dr3_source_id']
                target = r[r['SOURCE_ID']==row['dr3_source_id']]
                target_pos = SkyCoord(ra=target['ra'][0]*u.deg, dec=target['dec'][0]*u.deg, frame='icrs',location=self.world.telescope.position)
                ltt_bary = self.world.time_astropy.light_travel_time(target_pos)
                time_barycentre = t_tdb_jd + ltt_bary 

                impact_b_km = row['impact_b_km']
                r_ast_m = row['r_ast_m']
                d_obs_m = row['d_obs_m']
                theta_star_mas = row['theta_star_mas']
                v_rel_kms = row['v_rel_kms']
                bright_ast_frac = row['bright_ast_frac']
                wavelength_a = row['wavelength_a']
                rho_geom_limit = row['rho_geom_limit']
                reference_time = row['reference_time']  ##中心重合的时间
                MODE = row['MODE']

                # 计算时间差和位置
                time_diff = time_barycentre.jd - reference_time
                dt_seconds = time_diff * 86400  
                v_rel_mps = v_rel_kms * 1e3
                x_position = dt_seconds * v_rel_mps

                # 计算恒星半径
                star_rad = d_obs_m * (theta_star_mas / 1000 * np.pi / 648000) / 2

                # 计算rho参数决定模式
                F  = lambda l: np.sqrt(d_obs_m*l/2) # Fresnel scale (m)
                rho  = lambda l: r_ast_m/F(l)

                if MODE == "mono":
                    lam_set, wts = np.array([wavelength_a*1e-10]), np.array([1.0])  # wavelength  and weight
                    lam_eff = lam_set[0]; label_band = f"λ={wavelength_a:.0f} Å"  # effective wavelength
                else:
                    csv_path = self.filter
                    band = self.band_name
                    df = pd.read_csv(csv_path)
                    sub = df[(df.band==band)&(df.throughput_frac>0)]
                    if sub.empty: sys.exit(f"[ERROR] No throughput for '{band}'.")
                    lam_set = sub.wavelength.values*1e-10; wts = sub.throughput_frac.values; wts/=wts.sum()
                    lam_eff = np.sum(lam_set*wts); label_band = f"{band} band"
               
                KIND = "geometry" if rho(lam_eff) > rho_geom_limit else "diffraction"

                # 计算光通量
                if KIND == "geometry":
                    # 几何模式
                    impact_b_m = impact_b_km * 1e3
                    distance_to_center = np.hypot(x_position, impact_b_m)
                    coverage = overlap_area_circles(star_rad, r_ast_m, distance_to_center) / (np.pi * star_rad**2)
                    flux = 1 - (1 - bright_ast_frac) * coverage
                elif KIND == "diffraction":
                    # 衍射模式
                    impact_b_m = impact_b_km * 1e3
                    # r_position = np.hypot(x_position, impact_b_m)
                    flux = 0
                    for l,w in zip(lam_set,wts):
                        intensity = I_fresnel_diffraction(F, l, x_position, r_ast_m)  #直接用x_position衍射模式一般对应全遮掩
                        flux += intensity * w

                occultation_relative_flux[np.where(r['SOURCE_ID']==row['dr3_source_id'])] = flux

                new_occultation_data.append({
                'target_id': target_id,
                'tdb_jd': t_tdb_jd,
                'tdb_bjd': time_barycentre.jd,
                'normalized_flux': flux
                })
            if new_occultation_data:
                csv_occultation_path = Path(self.world.target_dir) / 'occultation_inputdata_sorted.csv'
                new_df = pd.DataFrame(new_occultation_data)
                if csv_occultation_path.exists() and csv_occultation_path.stat().st_size > 0:
                    try:
                        existing_df = pd.read_csv(csv_occultation_path)
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        combined_df = new_df
                else:
                    combined_df = new_df
                combined_df = combined_df.drop_duplicates(subset=['target_id', 'tdb_jd'], keep='last') ## drop duplicates (based on a combination of target_id and time, keeping the latest flux value)
                combined_df = combined_df.sort_values(['target_id', 'tdb_jd'])
                combined_df = combined_df.reset_index(drop=True)
                combined_df.to_csv(csv_occultation_path, index=False)

        return occultation_relative_flux

    def historical_variable_star(self, t_tdb_jd, r, transit_catalog, binary_catalog, flare_catalog, r_object_shape):
        if self.historical_variable_star_flag == False:
            return np.ones(r_object_shape)
        else:  
            Gmag_limit = self.Gmag_limit
            target_key = self.target_key

            def add_star_page_to_pdf(pdf, text, lc=None, time=None, flux=None, time_ref=None, flux_ref=None, period=None, t0=None):    
                fig = plt.figure(figsize=(8.5, 11))
                gs = gridspec.GridSpec(4, 1, height_ratios=[1, 2, 2, 2])

                # base information of star
                ax0 = fig.add_subplot(gs[0])
                ax0.axis("off")
                ax0.text(0, 1, text, va='top', fontsize=11, wrap=True)

                # light curve in lightkurve
                if lc is not None:
                    ax1 = fig.add_subplot(gs[1])
                    ax1.plot(lc.time.value, lc.flux.value, 'o', markersize=2)
                    ax1.set_title("Light Curve in Lightkurve (one sector)")
                    ax1.set_xlabel("Time (HJD)")
                    ax1.set_ylabel("Flux")

                # original light curve in one sector
                if time is not None and flux is not None:
                    ax2 = fig.add_subplot(gs[2])
                    ax2.plot(time, flux, 'ro', markersize=2, label='Original')
                    ax2.plot(time_ref, flux_ref, 'g.', alpha=0.3, markersize=2, label='LS')
                    ax2.legend()
                    ax2.set_title("A sector Light Curve")
                    ax2.set_xlabel("Time(HJD)")
                    ax2.set_ylabel("Flux")

                    cycle_number = np.floor((time - t0) / period)  # np.floor round down
                    unique_cycles, counts = np.unique(cycle_number, return_counts=True)  # np.unique all unique numbers of cycle
                    top1_cycles = unique_cycles[np.argsort(counts)[-1:]]  # [-1:] the most cycle
                    mask = np.isin(cycle_number, top1_cycles)  
                    time_cut = time[mask]
                    flux_cut = flux[mask]
                    idx = np.where((time_ref <= time_cut.max()) & (time_ref >= time_cut.min()))
                    time_cut_ref = time_ref[idx]
                    flux_cut_ref = flux_ref[idx]

                    ax3 = fig.add_subplot(gs[3])
                    ax3.plot(time_cut, flux_cut, 'ro', markersize=2, label='Original')
                    ax3.plot(time_cut_ref, flux_cut_ref, 'g.', alpha=0.3, markersize=2, label='LS')
                    ax3.legend()
                    ax3.set_title("Zoom in Light Curve")
                    ax3.set_xlabel("Time(HJD)")
                    ax3.set_ylabel("Flux")
                else:
                    fig.delaxes(fig.add_subplot(gs[2]))  # don't exist then remove
                    fig.delaxes(fig.add_subplot(gs[3])) 

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)    


            def query_vsx_in_region(ra, dec, radius):
                vizier = Vizier()
                vizier.ROW_LIMIT = -1  
                vizier.columns = ['Name', 'RAJ2000', 'DEJ2000', 'Type', 'Period', 'HJD']  
                coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
                result = vizier.query_region(
                    coord, 
                    radius = radius*u.deg, 
                    catalog = 'B/vsx/vsx', 
                    column_filters = {
                        'Gmag': f'<{Gmag_limit}',
                        # 'Period': 'NOT NULL',
                        # 'HJD': 'NOT NULL'
                        }
                )
                if result:
                    # print(result[0])
                    stars_T_P = []
                    stars_no_TP = []
                    for row in result[0]:
                        if row['Period'] == row['Period'] and row['Epoch'] == row['Epoch']:
                            stars_T_P.append((row['Name'], row['RAJ2000'], row['DEJ2000'], row['Type'], row['Period'], row['Epoch']))
                        else:
                            stars_no_TP.append((row['Name'], row['RAJ2000'], row['DEJ2000']))
                    return stars_T_P, stars_no_TP

            def get_lightcurve(target_coord):
                search_results = lk.search_lightcurve(target_coord, radius=0.0001 * u.deg)
                if search_results:
                    mask1 = np.char.find(search_results.author, "SPOC") >= 0 
                    mask2 = np.char.find(search_results.author, "Kepler") >= 0
                    mask3 = np.char.find(search_results.author, "K2") >= 0
                    mask = mask1 | mask2 | mask3
                    search_result = search_results[mask]
                    print(f"search_result in SPOC/Kepler/K2: {search_result}")
                    if search_result:
                        search_result = search_result[np.where(search_result.exptime == search_result.exptime.min())]
                        search_result = search_result[np.where(search_result.year == search_result.year.max())]
                        lc_collection = search_result.download_all()
                        lc = lc_collection[0]
                        mask_quality = (lc['quality'] == 0)
                        lc = lc[mask_quality]
                        if "TESS" in lc.mission:
                            lc.time = lc.time + 2457000.0
                            if "pdcsap_flux" in lc.columns and "pdcsap_flux_err" in lc.columns:
                                lc.flux = lc["pdcsap_flux"]
                                lc.flux_err = lc["pdcsap_flux_err"]  
                                print("use pdcsap_flux and pdcsap_flux_err")
                            elif "flux" in lc.columns and "flux_err" in lc.columns:
                                lc.flux = lc["flux"]
                                lc.flux_err = lc["flux_err"]  
                                print("use flux and flux_err")
                            else:
                                raise ValueError("lack effective flux or flux_err ")
                            lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

                        elif "Kepler" in lc.mission or "K2" in lc.mission:
                            lc.time = lc.time + 2454833.0
                            if "pdcsap_flux" in lc.columns and "pdcsap_flux_err" in lc.columns:
                                lc.flux = lc["pdcsap_flux"]
                                lc.flux_err = lc["pdcsap_flux_err"] 
                                print("use pdcsap_flux and pdcsap_flux_err")
                            elif "flux" in lc.columns and "flux_err" in lc.columns:
                                lc.flux = lc["flux"]
                                lc.flux_err = lc["flux_err"] 
                                print("use flux and flux_err")
                            else:
                                raise ValueError("lack effective flux or flux_err ")
                            lc = lc.remove_nans().remove_outliers(sigma=5).normalize()
                        return lc  
                    else:
                        ## after slect, only have QLP or other telescope(no kepler or K2)
                        search_result = search_results[np.where(search_results.exptime == search_results.exptime.min())]
                        search_result = search_results[np.where(search_results.year == search_results.year.max())]
                        lc_collection = search_result.download_all()
                        lc = lc_collection[0]
                        mask_quality = (lc['quality'] == 0)
                        lc = lc[mask_quality]
                        if "TESS" in lc.mission:
                            lc.time = lc.time + 2457000.0
                            if "sys_rm_flux" in lc.columns and "sys_rm_flux_err" in lc.columns:
                                lc.flux = lc["sys_rm_flux"]
                                lc.flux_err = lc["sys_rm_flux_err"] 
                                print("use sys_rm_flux and sys_rm_flux_err")
                            # elif "det_flux" in lc.columns and "det_flux_err" in lc.columns:
                            #     lc.flux = lc["det_flux"]
                            #     lc.flux_err = lc["det_flux_err"] 
                            #     print("use det_flux and det_flux_err")
                            #(det has a bad result)
                            elif "kspsap_flux" in lc.columns and "kspsap_flux_err" in lc.columns:
                                lc.flux = lc["kspsap_flux"]
                                lc.flux_err = lc["kspsap_flux_err"]  
                                print("use kspsap_flux and kspsap_flux_err")
                            else:
                                lc.flux = lc["flux"]
                                lc.flux_err = lc["flux_err"] 
                                print("use flux and flux_err")
                            lc = lc.remove_nans().remove_outliers(sigma=5).normalize()
                            plt.figure()
                            plt.plot(lc.time.value, lc.flux.value, '.')
                            plt.show()
                        # may have other telescope, need to consider then
                        
                        return lc 
                else:
                    return None

            def flux_subtraction_frequency(time, flux, p, params, fap_thresh=0.01):
                ls = LombScargle(time, flux, fit_mean=True, center_data=False)
                f_max = 1 / (p/12)
                f_min = 1 / 40
                frequency, power = ls.autopower(maximum_frequency=f_max, minimum_frequency=f_min)
                idx_max = np.argmax(power)
                freqs_max = frequency[idx_max]
                power_max = power[idx_max]
                fap = ls.false_alarm_probability(power_max)
                if fap < fap_thresh:
                    theta0, theta1, theta2 = ls.model_parameters(freqs_max)
                    A = np.hypot(theta1, theta2)
                    phi = np.arctan2(theta2, theta1)
                    params.append((freqs_max, A, phi, theta0))
                    flux_freqs = A * np.sin(2 * np.pi * freqs_max * time + phi) + theta0
                    flux_sub = flux - flux_freqs
                    return flux_sub, params
                else:
                    return None, params

            def process_lightcurve(lc, p, t0):
                time_origin = lc.time.value
                flux_origin = lc.flux.value
                params = []
                flux_sub, params = flux_subtraction_frequency(time_origin, flux_origin, p, params)
                while flux_sub is not None:
                    flux = flux_sub
                    flux_sub, params = flux_subtraction_frequency(time_origin, flux, p, params)

                time_ref = np.linspace(time_origin.min(), time_origin.max(), 100000)
                flux_ref = np.zeros_like(time_ref)

                for i in range(len(params)):
                    f = params[i][0]
                    A = params[i][1]
                    phi = params[i][2]
                    theta0 = params[i][3]

                    flux_ref += (A * np.sin(2 * np.pi * f * time_ref + phi) + theta0)

                return params, time_ref, flux_ref, time_origin, flux_origin

            def get_flux_at_time(params, time):
                flux = 0
                for i in range(len(params)):
                    f = params[i][0]
                    A = params[i][1]
                    phi = params[i][2]
                    theta0 = params[i][3]
                    flux += (A * np.sin(2 * np.pi * f * time + phi) + theta0)
                return flux

            ra = Angle(self.world.telescope.mount.ra_deg, unit='deg')
            dec = Angle(self.world.telescope.mount.dec_deg, unit='deg')
            # radius = self.world.telescope.fov_diag/2
            print(self.world.telescope.fov_diag/2)
            vsx_relative_flux = np.ones(r_object_shape)
            new_variable_data = []
            if 'target_key' in photons_distribution_simulator.vsx_dict and target_key == photons_distribution_simulator.vsx_dict['target_key']:
               vsx_dict = photons_distribution_simulator.vsx_dict
               stars = vsx_dict['star']
               find_id_result = vsx_dict['id']
               parameters = vsx_dict['parameters']
               print("stars from vsx_dict:", stars, "find_id_result:", find_id_result, sep='\n')
               for i in range(len(stars)):
                    name = stars[i][0]
                    star_ra = stars[i][1]
                    star_dec = stars[i][2]
                    types = stars[i][3]
                    period = stars[i][4]
                    t0 = stars[i][5]
                    print(f"star{i}:{stars[i]}, star_ra:{star_ra}, star_dec:{star_dec}, tpye:{types}, period:{period}, t0:{t0}")
                    target_pos = SkyCoord(ra=star_ra*u.deg, dec=star_dec*u.deg, frame='icrs',location=self.world.telescope.position)
                    ltt_bary = self.world.time_astropy.light_travel_time(target_pos)
                    time_barycentre = t_tdb_jd + ltt_bary
                    gr3_id = vsx_dict['id'][i]
                    parameter = vsx_dict['parameters'][i]
                    time_query = time_barycentre.jd
                    vsx_normalized_flux = get_flux_at_time(parameter, time_query)
                    # relative pixcrd to simply find the star, if not:
                    # vsx_relative_flux[np.where(r['SOURCE_ID']==gr3_id)] = vsx_normalized_flux
                    mask_vsx = np.where(r['SOURCE_ID']==gr3_id)
                    vsx_relative_flux[mask_vsx] = vsx_normalized_flux
                    relative_ra = r['ra'][mask_vsx]
                    relative_dec = r['dec'][mask_vsx]
                    relative_radec_star = np.column_stack((relative_ra, relative_dec))
                    relative_pixcrd = self.world.telescope.wcs.wcs_world2pix(relative_radec_star, 0)
                    print(f"gr3_id:{gr3_id}, flux:{vsx_normalized_flux}")

                    new_variable_data.append({
                        'name': name,
                        'Gaia DR3 id': gr3_id,
                        'type': types,
                        'tdb_jd': t_tdb_jd,
                        'tdb_bjd': time_barycentre.jd,
                        'flux': vsx_normalized_flux,
                        'pixcrd': relative_pixcrd,
                        'parameter': parameter
                    })

            else:
                stars_T_P, stars_no_TP = query_vsx_in_region(ra, dec, self.world.telescope.fov_diag/2)

                events_catalog_list = [transit_catalog, binary_catalog, flare_catalog]
                excluded_ids = set()
                for cat_path in events_catalog_list:
                    df = pd.read_csv(cat_path)  
                    excluded_ids.update(df["dr3_source_id"].dropna().tolist())

                if stars_T_P:
                    select_satrs = []
                    id = []
                    parameters = []
                    ## the target_key may be replaced by target_name
                    pdf_filename = Path(self.world.target_dir) / f"variable_LS_output_{self.world.schedule_target_name}.pdf"
                    pp = PdfPages(pdf_filename)
                    output_buffer = StringIO()

                    # Gaia catalog coords
                    gaia_ra = np.array(r['ra'])
                    gaia_dec = np.array(r['dec'])
                    gaia_coords = np.vstack((gaia_ra, gaia_dec)).T  # shape: (N, 2)

                    # stars_T_P coords
                    star_coords = np.array([[s[1], s[2]] for s in stars_T_P])  # shape: (M, 2)

                    tree = KDTree(gaia_coords)
                    dist, gaia_idx = tree.query(star_coords, distance_upper_bound=(5.0 / 3600))  # 5 arcsec in degrees


                    for i, (star_name, star_ra, star_dec, types, period, t0) in enumerate(stars_T_P):
                        print(f"{i} Star(search in vsx): {star_name}\nRA: {star_ra} Dec: {star_dec}\nType: {types} Period: {period} days T0: {t0}\n")
                        text = f"{i} Star: {star_name}\nRA: {star_ra} Dec: {star_dec}\nType: {types} Period: {period} days T0: {t0}\n"
                        output_buffer.write(text + '\n')

                        target_pos = SkyCoord(ra=star_ra*u.deg, dec=star_dec*u.deg, frame='icrs',location=self.world.telescope.position)
                        ltt_bary = self.world.time_astropy.light_travel_time(target_pos)
                        time_barycentre = t_tdb_jd + ltt_bary
                        time_query = time_barycentre.jd

                        target_coord = SkyCoord(star_ra, star_dec, unit=(u.deg, u.deg), frame='icrs')
                        d = dist[i]

                        if np.isfinite(d):
                            gr3_id = r['SOURCE_ID'][gaia_idx[i]]
                            print(f"gr3_id:{gr3_id}")
                            if gr3_id not in excluded_ids:
                                lc_collection = get_lightcurve(target_coord)
                                if lc_collection:
                                    if period <= 27.4:
                                        parameter, phase_ref, flux_ref, phase_origin, flux_origin = process_lightcurve(lc_collection, period, t0)
                                        vsx_normalized_flux = get_flux_at_time(parameter, time_query)
                                        print(f"{star_name}, {gr3_id}: Flux at time {time_query}: {vsx_normalized_flux}\n")
                                        add_star_page_to_pdf(pp, output_buffer.getvalue(), lc_collection, phase_origin, flux_origin, phase_ref, flux_ref, period, t0)
                                        # Empty the buffer
                                        output_buffer.truncate(0)
                                        output_buffer.seek(0)
                                        if vsx_normalized_flux != 0: 
                                            mask_vsx = np.where(r['SOURCE_ID']==gr3_id)
                                            vsx_relative_flux[mask_vsx] = vsx_normalized_flux
                                            relative_ra = r['ra'][mask_vsx]
                                            relative_dec = r['dec'][mask_vsx]
                                            relative_radec_star = np.column_stack((relative_ra, relative_dec))
                                            relative_pixcrd = self.world.telescope.wcs.wcs_world2pix(relative_radec_star, 0)
                                            select_satrs.append([star_name, star_ra, star_dec, types, period, t0])
                                            id.append(gr3_id)
                                            parameters.append(parameter)

                                            new_variable_data.append({
                                                'name': star_name,
                                                'Gaia DR3 id': gr3_id,
                                                'type': types,
                                                'tdb_jd': t_tdb_jd,
                                                'tdb_bjd': time_barycentre.jd,
                                                'flux': vsx_normalized_flux,
                                                'pixcrd': relative_pixcrd,
                                                'parameter': parameter
                                            })

                                    else:
                                        add_star_page_to_pdf(pp, output_buffer.getvalue(), lc_collection)
                                        output_buffer.truncate(0)
                                        output_buffer.seek(0)
                                        print("Period is too long, skipping this star.") 
                                else:
                                    add_star_page_to_pdf(pp, output_buffer.getvalue())
                                    output_buffer.truncate(0)
                                    output_buffer.seek(0)

                    pp.close()    

                    if new_variable_data:
                        csv_variable_path = Path(self.world.target_dir) / 'variable_inputdata_sorted.csv'
                        new_df = pd.DataFrame(new_variable_data)
                        if csv_variable_path.exists() and csv_variable_path.stat().st_size > 0:
                            try:
                                existing_df = pd.read_csv(csv_variable_path)
                                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                                combined_df = new_df
                        else:
                            combined_df = new_df
                        combined_df = combined_df.drop_duplicates(subset=['name', 'tdb_jd'], keep='last') ## drop duplicates (based on a combination of target_id and time, keeping the latest flux value)
                        combined_df = combined_df.sort_values(['name', 'tdb_jd'])
                        combined_df = combined_df.reset_index(drop=True)
                        combined_df.to_csv(csv_variable_path, index=False)        

                    photons_distribution_simulator.update_vsx_data_dict(target_key, select_satrs, id, parameters)

                # stars_no_TP will update later
                else:
                    print("No periodic variable stars found in the given region using vsx search.")

            return vsx_relative_flux   

    def flux_variable_all_photons(self, img):
        
        star_catalog = self.star_catalog
        Gmag_limit = self.Gmag_limit
        alpha = self.alpha 
        transit_catalog = self.transit_catalog
        binary_catalog = self.binary_catalog
        flare_catalog = self.flare_catalog
        occultation_catalog = self.occultation_catalog
        
        
        target_key = self.target_key
        if 'target_key' in photons_distribution_simulator.gaia_dict and target_key == photons_distribution_simulator.gaia_dict['target_key']:

            r = photons_distribution_simulator.gaia_dict['gaia_data']
        elif star_catalog == 'online':
            # Get the gaia source that is not galaxy
            sql = f'''
            SELECT g3.source_id as source_id,g3.ra,g3.dec,g3.phot_g_mean_mag,g3.phot_g_mean_flux_error,g3.phot_g_n_obs,g3.phot_g_mean_flux, g3.phot_variable_flag,g3.parallax,g3.pmra,g3.pmdec from gaiadr3.gaia_source as g3 LEFT JOIN 	
gaiadr3.galaxy_candidates as ggc ON ggc.source_id = g3.source_id
WHERE g3.phot_g_mean_mag<{Gmag_limit} AND
CONTAINS(
    POINT('ICRS',g3.ra,g3.dec),
    CIRCLE('ICRS',{self.world.telescope.mount.ra_deg},{self.world.telescope.mount.dec_deg},{self.world.telescope.fov_diag/2})
)=1 AND ggc.source_id IS NULL AND g3.phot_g_mean_mag=g3.phot_g_mean_mag;'''      
            job = Gaia.launch_job_async(sql)
            r = job.get_results()
            photons_distribution_simulator.update_gaia_data_dict(target_key, r)
        else:
            radius_deg = self.world.telescope.fov_diag / 2 
            r = chunked_star_search(
                fits_path=star_catalog,
                ra0=self.world.telescope.mount.ra_deg,
                dec0=self.world.telescope.mount.dec_deg,
                radius_deg=radius_deg,
                gmag_limit=Gmag_limit,
                chunk_size=5000000,
                memory_limit_gb=2
            )
            photons_distribution_simulator.update_gaia_data_dict(target_key, r)

        mag_raw = r['phot_g_mean_mag']

        # Get the magnitude error
        if star_catalog == 'online':
            error_fraction_jitter_sample = self.jitter_error(r)
            flux = self.T_lambda  * 1e10 * 1.346109e-21 * r['phot_g_mean_flux']
        else:
            error_fraction_jitter_sample = np.ones(len(r)) # do not apply jitter error due to the lack columns of local catalog
            flux = self.T_lambda * 1e-3 * 10**(-0.4*(mag_raw-self.zero_mag))
        luminosity_fraction_extinction = self.extinction_error(r['ra'], r['dec'])

        t_tdb_jd = self.world.time_astropy.tdb.jd

        # Get the coordinates error of ADR
        if self.ADR_flag == True:
            error_ADR_coordinates = self.ADR(t_tdb_jd, np.array(r['ra']), np.array(r['dec']))
            r['ra'] += error_ADR_coordinates[0]
            r['dec'] += error_ADR_coordinates[1]

        # Consider the events of transits
        transit_relative_flux = self.transit(t_tdb_jd, transit_catalog, r, mag_raw.shape)
        # Consider the events of binary
        binary_relative_flux = self.binary(t_tdb_jd, binary_catalog, r, mag_raw.shape)      
        # consider the events of flare
        flare_relative_flux = self.flare(t_tdb_jd, flare_catalog, r, mag_raw.shape)
        # consider the events of occultation
        occultation_relative_flux = self.occultation(t_tdb_jd, occultation_catalog, r, mag_raw.shape)
        # consider the variable stars without the physical parameters
        # use LombScargle to the light curve from the TESS/Kepler data 
        vsx_relative_flux = self.historical_variable_star(t_tdb_jd, r, transit_catalog, binary_catalog, flare_catalog, mag_raw.shape)


        flux_prod = transit_relative_flux * flare_relative_flux * binary_relative_flux * occultation_relative_flux * vsx_relative_flux * error_fraction_jitter_sample * luminosity_fraction_extinction

        hnu = self.hnu
        FWHM = self.FWHM
        gamma = self.gamma

        # normalize_factor =  (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s  * (alpha - 1)/ ( gamma**2 * hnu)
        n_photon =  flux * flux_prod
        # normalize_factor * 
        image_star = np.zeros(img.shape)
        radec_star = np.array([[r['ra'][i],r['dec'][i]] for i in range(len(r))])
        pixcrd = self.world.telescope.wcs.wcs_world2pix(radec_star, 0)
        # output_filename = "/Users/kexin_li/sky_maker/wasp_11_pixcrd.list"
        # with open(output_filename, 'w') as f:
        #     # 写入文件头注释
        #     f.write("# object list file\n")
        #     f.write("# Format: <code (100=star, 200=galaxy)> <x> <y> <magnitude> <...>\n")
        #     f.write("# Generated from stellar catalog data\n")
        #     f.write("#\n")

        #     # 遍历所有恒星
        #     for i in range(len(r)):
        #         # 获取像素坐标
        #         x_pix = pixcrd[i][0]
        #         y_pix = pixcrd[i][1]
        #         final_magnitude = r['phot_g_mean_mag'][i]
        #         f.write(f"100 {x_pix:.3f} {y_pix:.3f} {final_magnitude:.3f}\n")
        # print(f"WASP list file '{output_filename}' generated successfully.")
        # print(f"Total stars in field of view: {i+1}")
        moffat_pixel =  FWHM * self.moffat_scale_FWHM

        x0 = [item[0] for item in pixcrd]
        y0 = [item[1] for item in pixcrd]
        for x0, y0, A in zip(x0, y0, n_photon):
            x_min = max(int(x0 - moffat_pixel), 0)
            x_max = min(int(x0 + moffat_pixel), img.shape[1])
            y_min = max(int(y0 - moffat_pixel), 0)
            y_max = min(int(y0 + moffat_pixel), img.shape[0])
            if x_min >= x_max or y_min >= y_max:
                continue  
            X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            lower = self.FWHM*(1-3*self.FWHM_error)
            upper = self.FWHM*(1+3*self.FWHM_error)
            FWHM_each = np.clip(np.random.normal(loc=self.FWHM, scale=self.FWHM_error * self.FWHM),
                     a_min=lower, a_max=upper)
            gamma_each = FWHM_each / (2 * np.sqrt(2**(1/self.alpha) - 1))
            normalize_factor_each =  (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s * (self.alpha - 1)/ (gamma_each**2 * self.hnu) 
            A_each = A * normalize_factor_each
            g = Moffat2D(amplitude=A_each, x_0=x0, y_0=y0, gamma=gamma_each, alpha=self.alpha)
            image_star[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)

        return image_star

    # consider coordinate_variable stars
    def satellite(self, img):
        image_satellite = np.zeros(img.shape)
        satellite_flag = self.satellite_flag
        satellite_catalog = self.satellite_catalog
        if satellite_flag:
            ts = load.timescale()
            hjd0 = self.world.time_astropy.tdb.jd
            hjd1 = hjd0 + self.world.telescope.camera.exposure_s / 86400   # in unit of day
            t0 = ts.tt_jd(hjd0)
            t1 = ts.tt_jd(hjd1)
            
            def satellite_intersection(x1, y1, x2, y2, img_width, img_height) :
                rect_x_min, rect_y_min = 0, 0
                rect_x_max, rect_y_max = img_width - 1, img_height - 1

                # Quick reject: if both points are on the same side outside the image
                if ((x1 < rect_x_min and x2 < rect_x_min) or 
                    (x1 > rect_x_max and x2 > rect_x_max) or
                    (y1 < rect_y_min and y2 < rect_y_min) or
                    (y1 > rect_y_max and y2 > rect_y_max)):
                    return False

                # Fast acceptance: if any point is within the image
                if ((rect_x_min <= x1 <= rect_x_max and rect_y_min <= y1 <= rect_y_max) or
                    (rect_x_min <= x2 <= rect_x_max and rect_y_min <= y2 <= rect_y_max)):
                    return True

                # Using a simplified version of the Cohen-Sutherland algorithm
                dx = x2 - x1
                dy = y2 - y1
                t_values = []

                if abs(dx) > 1e-10:
                    t_left = (rect_x_min - x1) / dx
                    t_right = (rect_x_max - x1) / dx
                    t_values.extend([t_left, t_right])

                if abs(dy) > 1e-10:
                    t_bottom = (rect_y_min - y1) / dy
                    t_top = (rect_y_max - y1) / dy
                    t_values.extend([t_bottom, t_top])

                # Checking valid intersections
                for t in t_values:
                    if 0 <= t <= 1:  # make sure intersection is inside the line
                        x = x1 + t * dx
                        y = y1 + t * dy
                        if (rect_x_min <= x <= rect_x_max and rect_y_min <= y <= rect_y_max):
                            return True

                return False




            # tle_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'  # latest active satellite, but we need t0
            ts = load.timescale()
            eph = load('de421.bsp') 
            # satellites = load.tle_file(tle_url)
            ts = load.timescale()
            tle_filename = satellite_catalog
            with load.open(tle_filename) as f:
                lines = list(parse_tle_file(f, ts))
            
            for satellite in lines:
                ra0, dec0, _ = satellite.at(t0).radec()
                ra1, dec1, _ = satellite.at(t1).radec()
                ra0_deg  = ra0._degrees # ra use hour not degree
                dec0_deg = dec0.degrees
                ra1_deg  = ra1._degrees
                dec1_deg = dec1.degrees  # change numpy to units.Angle 

                # make sure it's numpy array and float48
                coords0 = np.array([[ra0_deg, dec0_deg]], dtype=np.float64)
                coords1 = np.array([[ra1_deg, dec1_deg]], dtype=np.float64)
        
                pixcrd0 = self.world.telescope.wcs.wcs_world2pix(coords0, 0)
                pixcrd1 = self.world.telescope.wcs.wcs_world2pix(coords1, 0)

                sunlit = satellite.at(t0+(t1-t0)/2).is_sunlit(eph)
                # astropy.time.Time 对象不能直接相加，但可以与 astropy.time.TimeDelta 对象相加
                if sunlit:
                    if satellite_intersection(pixcrd0[0][0], pixcrd0[0][1], pixcrd1[0][0], pixcrd1[0][1], img.shape[1], img.shape[0]):  
                        x1, y1, x2, y2 = int(pixcrd0[0][0]), int(pixcrd0[0][1]), int(pixcrd1[0][0]), int(pixcrd1[0][1]) 
                        num = max(abs(x2 - x1), abs(y2 - y1)) + 1
                        num = int(num * self.move_oversample_factor)
                        x_list = np.linspace(x1, x2, num=num, dtype=float)
                        y_list = np.linspace(y1, y2, num=num, dtype=float)
                        pixcrd_satellite = list(zip(x_list, y_list)) 
                        # pixcrd_satellite = bresenham_line(int(pixcrd0[0][0]), int(pixcrd0[0][1]), int(pixcrd1[0][0]), int(pixcrd1[0][1]))  # the result like[(0, 0), ()], a list
                        time_each_pixel = self.world.telescope.camera.exposure_s / len(pixcrd_satellite)
                        flux_satellite = np.zeros(len(pixcrd_satellite)) + self.T_lambda * 1e-3 * 10**(-0.4*(self.satellite_mag-self.zero_mag)) 

                        print(pixcrd_satellite)
                        print(f"satellite flux: {flux_satellite}")

                        # normalize_factor_satellite =  (self.world.telescope.diameter_m/2)**2 * time_each_pixel * (self.alpha - 1)/ (self.gamma**2 * self.hnu)
                        n_photon =  flux_satellite  # error don't consider now
                        # normalize_factor_satellite * 
                        moffat_pixel = self.moffat_scale_FWHM * self.FWHM
                       
                        x0 = [item[0] for item in pixcrd_satellite]
                        y0 = [item[1] for item in pixcrd_satellite]
                        for x0, y0, A in zip(x0, y0, n_photon):
                            x_min = max(int(x0 - moffat_pixel), 0)
                            x_max = min(int(x0 + moffat_pixel), img.shape[1])
                            y_min = max(int(y0 - moffat_pixel), 0)
                            y_max = min(int(y0 + moffat_pixel), img.shape[0])
                            if x_min >= x_max or y_min >= y_max:
                                continue  
                            X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
                            lower = self.FWHM*(1-3*self.FWHM_error)
                            upper = self.FWHM*(1+3*self.FWHM_error)
                            FWHM_each = np.clip(np.random.normal(loc=self.FWHM, scale=self.FWHM_error * self.FWHM),
                                     a_min=lower, a_max=upper)
                            gamma_each = FWHM_each / (2 * np.sqrt(2**(1/self.alpha) - 1))
                            normalize_factor_each =  (self.world.telescope.diameter_m/2)**2 * time_each_pixel * (self.alpha - 1)/ (gamma_each**2 * self.hnu)
                            A_each = A * normalize_factor_each
                            g = Moffat2D(amplitude=A_each, x_0=x0, y_0=y0, gamma=gamma_each, alpha=self.alpha)
                            # Calculate the distance from the center at each coordinate and then calculate the flux value using the parameter
                            image_satellite[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)    

        return image_satellite

    def asteroid(self, img):
        image_asteroid = np.zeros(img.shape)
        asteroid_flag = self.asteroid_flag

        if asteroid_flag:
            obs = self.world.time_astropy.tdb.jd

            url = "https://ssd-api.jpl.nasa.gov/sb_ident.api"
            ra_center = Angle(self.world.telescope.mount.ra_deg, unit='deg')
            dec_center = Angle(self.world.telescope.mount.dec_deg, unit='deg')
            ra_str  = ra_center.to_string(unit=u.hour, sep='-', precision=2, pad=True)  # e.g. "16-48-45.00"
            dec_str = dec_center.to_string(unit=u.deg,  sep='-', precision=2, alwayssign=True, pad=True)  # e.g. "+21-23-01.00"
            ra_fov_limt = self.world.telescope.fov_diag/2
            dec_fov_limt = self.world.telescope.fov_diag/2
            query_params = {
                'sb-kind': 'a',  # Limit results to either asteroids-only (a) or comets-only (c)
                "obs-time": obs,  # UC/jd
                # 'mpc-code': 'O18',
                'lon': tel.latlonalt[1],  # Longitude
                'lat': tel.latlonalt[0],  # Latitude
                'alt': tel.latlonalt[2],  # Altitude
                # 'sb-class': sb_class,
                "fov-ra-center": ra_str,  # hh:MM:SS
                "fov-dec-center": dec_str,  # dd:MM:SS
                "fov-ra-hwidth": ra_fov_limt,  # radius degrees
                "fov-dec-hwidth": dec_fov_limt,  
                "vmag-lim": self.Gmag_limit,  # limiting magnitude
                # 'fmt-ra-dec': 'false',
                # 'output-sort': 'trans',
                "mag-required": "true",  # Whether to skip objects without magnitude
                'two-pass': 'true',  # Enable second-pass filtering, which will use orbital propagation computation
                'suppress-first-pass': 'true',  # Suppress first pass data
                # The first pass is just a rough pre-screening to fall into the field of view using an approximation (spherical center coordinates) without orbit propagation
            }
            # Get the response
            response = requests.get(url, params=query_params,verify=False)
            raw_data = response.json()
            # Get the complete request URL
            website = response.url
            print(f"Request URL for asteroids: {website}")
            # Check if 'data_second_pass' or 'data_first_pass' exists
            if 'data_second_pass' in raw_data:
                # Use 'fields_second' as column names
                columns = raw_data.get('fields_second', [])
                data = raw_data['data_second_pass']
                asteroid_continue = True
            # elif 'data_first_pass' in raw_data:
            #     # Use 'fields_first' as column names
            #     columns = raw_data.get('fields_first', [])
            #     data = raw_data['data_first_pass']
            #     asteroid_continue = True
            else:
                asteroid_continue = False

            if asteroid_continue:
                new_asteroid_data = []

                df = pd.DataFrame(data, columns=columns)  # Convert the data to a DataFrame
                name_asteroid = df['Object name']
                ra_asteroid = Angle(df['Astrometric RA (hh:mm:ss)'], unit=u.hourangle).degree
                dec_ = df["Astrometric Dec (dd mm'ss\")"].str.replace('\'', ' ').str.replace('"', ' ')   
                dec_asteroid = Angle(dec_, unit=u.deg).degree
                ra_rate = df['RA rate ("/h)'].astype(float)
                dec_rate = df['Dec rate ("/h)'].astype(float)
                mag_V_asteroid = df['Visual magnitude (V)'].astype(float)
                tdb_jd_asteroid = [obs] * len(name_asteroid)

                move_arcsec = np.sqrt(np.array(ra_rate)**2 + np.array(dec_rate)**2) * (self.world.telescope.camera.exposure_s / 3600)  # in unit of arcsec
                move_pixel = move_arcsec / self.world.telescope.arcsec_pixel_1  # in unit of pixel

                FWHM = self.FWHM
                moffat_pixel = self.moffat_scale_FWHM * FWHM  

                # if move_pixel > self.asteroid_move_limit_pixel:
                #     ra_0, dec_0 = ra_asteroid, dec_asteroid
                #     ra_1 = ra_asteroid + ra_rate * (self.world.telescope.camera.exposure_s / 3600)
                #     dec_1 = dec_asteroid + dec_rate * (self.world.telescope.camera.exposure_s / 3600)

                # should consider the band transformation then, several image layers
                # mag_g_asteroid = mag_V_asteroid + 0.02266 - 0.27125 * 0.8 - 0.11207 * 0.8**2  # V-I = 0.8
                flux_asteroid = self.T_lambda * 1e-3 * 10**(-0.4*(np.array(mag_V_asteroid)-self.zero_mag)) # magnitude conversion between layers later
                # no magnitude error, because no photometric datas
                luminosity_fraction_extinction_asteroid = self.extinction_error(ra_asteroid, dec_asteroid)
                if self.ADR_flag:
                    error_ADR_coordinates_asteroid = self.ADR(obs, ra_asteroid, dec_asteroid)
                    ra_asteroid += error_ADR_coordinates_asteroid[0]
                    dec_asteroid += error_ADR_coordinates_asteroid[1]
                radec_star = np.array([[ra_asteroid[i],dec_asteroid[i]] for i in range(len(ra_asteroid))])
                pixcrd = self.world.telescope.wcs.wcs_world2pix(radec_star, 0)
                # 分离快速和慢速目标
                fast_mask = move_pixel > self.asteroid_move_limit_pixel
                print(f"找到 {len(name_asteroid)} 颗小行星: {np.sum(fast_mask)} 快速, {np.sum(~fast_mask)} 慢速")

                if np.sum(~fast_mask) > 0:
                    # normalize_factor_asteroid =  (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s  * (self.alpha - 1)/ ( self.gamma**2 * self.hnu)
                    n_photon = flux_asteroid[~fast_mask] * luminosity_fraction_extinction_asteroid[~fast_mask]
                    # normalize_factor_asteroid * 

                    x0 = [item[0] for item in pixcrd[~fast_mask]]
                    y0 = [item[1] for item in pixcrd[~fast_mask]]
                    for x0, y0, A in zip(x0, y0, n_photon):
                        x_min = max(int(x0 - moffat_pixel), 0)
                        x_max = min(int(x0 + moffat_pixel), img.shape[1])
                        y_min = max(int(y0 - moffat_pixel), 0)
                        y_max = min(int(y0 + moffat_pixel), img.shape[0])
                        if x_min >= x_max or y_min >= y_max:
                            continue  
                        X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
                        lower = self.FWHM*(1-3*self.FWHM_error)
                        upper = self.FWHM*(1+3*self.FWHM_error)
                        FWHM_each = np.clip(np.random.normal(loc=self.FWHM, scale=self.FWHM_error * self.FWHM),
                                 a_min=lower, a_max=upper)
                        gamma_each = FWHM_each / (2 * np.sqrt(2**(1/self.alpha) - 1))
                        normalize_factor_each =  (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s  * (self.alpha - 1)/ (gamma_each**2 * self.hnu) 
                        A_each = A * normalize_factor_each
                        g = Moffat2D(amplitude=A_each, x_0=x0, y_0=y0, gamma=gamma_each, alpha=self.alpha)
                        image_asteroid[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)

                    for i in np.where(~fast_mask)[0]:
                    # np.where(~fast_mask) 返回一个元组，其中包含了这些"慢速小行星"在数组中的索引。[0] 取出这个索引数组
                        new_asteroid_data.append({
                            'name_asteroid': name_asteroid.iloc[i],
                            'ra_asteroid': ra_asteroid[i],
                            'dec_asteroid': dec_asteroid[i],
                            'ra_rate': ra_rate.iloc[i], 
                            'dec_rate': dec_rate.iloc[i], 
                            'tdb_jd_asteroid': tdb_jd_asteroid[i], 
                            'mag_V_asteroid': mag_V_asteroid.iloc[i], 
                            'pixcrd_x': pixcrd[i][0],  # 分别保存x和y坐标
                            'pixcrd_y': pixcrd[i][1],
                            'mode': 'slow'                    
                        })

                if np.sum(fast_mask) > 0:
                    for i in np.where(fast_mask)[0]:
                        # 计算起点和终点
                        ra_0, dec_0 = ra_asteroid[i], dec_asteroid[i]
                        print(f'ra_rate: {ra_rate.iloc[i]}, dec_rate: {dec_rate.iloc[i]}')
                        ra_1 = ra_0 + ra_rate.iloc[i] * (self.world.telescope.camera.exposure_s / 3600) / 3600
                        dec_1 = dec_0 + dec_rate.iloc[i] * (self.world.telescope.camera.exposure_s / 3600) / 3600
                        print(f'起点: ({ra_0}, {dec_0}), 终点: ({ra_1}, {dec_1})')

                        # 转换为像素坐标
                        coord_0 = np.array([[ra_0, dec_0]], dtype=np.float64)
                        coord_1 = np.array([[ra_1, dec_1]], dtype=np.float64)
                        pixcrd_0 = self.world.telescope.wcs.wcs_world2pix(coord_0, 0)
                        pixcrd_1 = self.world.telescope.wcs.wcs_world2pix(coord_1, 0)
                        print(f'像素坐标起点: {pixcrd_0}, 终点: {pixcrd_1}')

                        x1, y1, x2, y2 = int(pixcrd_0[0][0]), int(pixcrd_0[0][1]), int(pixcrd_1[0][0]), int(pixcrd_1[0][1])

                        # 生成轨迹点（过采样）
                        num = max(abs(x2 - x1), abs(y2 - y1)) + 1
                        num *= self.move_oversample_factor
                        x_list = np.round(np.linspace(x1, x2, num=num, dtype=float))
                        y_list = np.round(np.linspace(y1, y2, num=num, dtype=float))
                        pixcrd_trail = list(zip(x_list, y_list))

                        # 计算每个点的参数
                        time_each_pixel = self.world.telescope.camera.exposure_s / len(pixcrd_trail)
                        flux_each = flux_asteroid[i] * luminosity_fraction_extinction_asteroid[i]
                        # normalize_factor_trail = ((self.world.telescope.diameter_m/2)**2 * 
                        #                          time_each_pixel * (self.alpha - 1) / 
                        #                          (self.gamma**2 * self.hnu))
                        n_photon = flux_each
                        # * normalize_factor_trail 

                        print(f"  {name_asteroid.iloc[i]}: 移动 {move_pixel[i]:.1f} 像素, "f"轨迹点数 {len(pixcrd_trail)}")
                        # 沿轨迹添加Moffat
                        for x0, y0 in pixcrd_trail:
                            x_min = max(int(x0 - moffat_pixel), 0)
                            x_max = min(int(x0 + moffat_pixel), img.shape[1])
                            y_min = max(int(y0 - moffat_pixel), 0)
                            y_max = min(int(y0 + moffat_pixel), img.shape[0])

                            if x_min >= x_max or y_min >= y_max:
                                continue
                            
                            X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
                            lower = self.FWHM*(1-3*self.FWHM_error)
                            upper = self.FWHM*(1+3*self.FWHM_error)
                            FWHM_each = np.clip(np.random.normal(loc=self.FWHM, scale=self.FWHM_error * self.FWHM),
                                     a_min=lower, a_max=upper)
                            gamma_each = FWHM_each / (2 * np.sqrt(2**(1/self.alpha) - 1))
                            normalize_factor_each =  (self.world.telescope.diameter_m/2)**2 * time_each_pixel  * (self.alpha - 1)/ (gamma_each**2 * self.hnu)
                            A_each = n_photon * normalize_factor_each
                            g = Moffat2D(amplitude=A_each, x_0=x0, y_0=y0,
                                       gamma=gamma_each, alpha=self.alpha)
                            image_asteroid[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)

                        # 保存数据
                        new_asteroid_data.append({
                            'name_asteroid': name_asteroid.iloc[i],
                            'ra_asteroid': ra_asteroid[i],
                            'dec_asteroid': dec_asteroid[i],
                            'ra_rate': ra_rate.iloc[i],
                            'dec_rate': dec_rate.iloc[i],
                            'tdb_jd_asteroid': tdb_jd_asteroid[i],
                            'mag_V_asteroid': mag_V_asteroid.iloc[i],
                            'pixcrd_x': pixcrd[i][0],
                            'pixcrd_y': pixcrd[i][1],
                            'mode': 'fast'
                        })

                
                if new_asteroid_data:
                    csv_asteroid_path = Path(self.world.target_dir) / 'asteroid_inputdata_sorted.csv'
                    new_df = pd.DataFrame(new_asteroid_data)
                    if csv_asteroid_path.exists() and csv_asteroid_path.stat().st_size > 0:
                        try:
                            existing_df = pd.read_csv(csv_asteroid_path)
                            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        except (pd.errors.EmptyDataError, pd.errors.ParserError):
                            combined_df = new_df
                    else:
                        combined_df = new_df
                    combined_df = combined_df.drop_duplicates(subset=['name_asteroid', 'tdb_jd_asteroid'], keep='last') ## drop duplicates (based on a combination of target_id and time, keeping the latest flux value)
                    combined_df = combined_df.sort_values(['name_asteroid', 'tdb_jd_asteroid'])
                    combined_df = combined_df.reset_index(drop=True)
                    combined_df.to_csv(csv_asteroid_path, index=False)

        return image_asteroid

        # galaxy 
    
    # consider galaxy and supernova erupt in galaxy
    def galaxy_supernova(self, img):
        if self.galaxy_flag == False:
            return np.zeros(img.shape)
        else:
            target_key = self.target_key
            image_gal = np.zeros(img.shape)
            image_supernova = np.zeros(img.shape)
            supernova_erupt_catalog = self.supernova_erupt_catalog
            gal_catalog = self.gal_catalog
            supernova_erupt = self.supernova_erupt
            if 'target_key' in photons_distribution_simulator.galaxy_sn_dict and target_key == photons_distribution_simulator.galaxy_sn_dict['target_key']:
                galaxy_sn_dict = photons_distribution_simulator.galaxy_sn_dict
                select_data = galaxy_sn_dict['select_data']
                print(f"select galaxy from dict: {select_data}")
            else:

                gal_data = pd.read_csv(gal_catalog)
                ra_min = self.world.telescope.mount.ra_deg - self.world.telescope.fov_diag/2/np.cos(self.world.telescope.mount.dec_deg*np.pi/180)
                ra_max = self.world.telescope.mount.ra_deg + self.world.telescope.fov_diag/2/np.cos(self.world.telescope.mount.dec_deg*np.pi/180)
                dec_min = self.world.telescope.mount.dec_deg - self.world.telescope.fov_diag/2
                dec_max = self.world.telescope.mount.dec_deg + self.world.telescope.fov_diag/2
                select_data = gal_data[
                    (gal_data['ra'] >= ra_min) & (gal_data['ra'] <= ra_max) & 
                    (gal_data['dec'] >= dec_min) & (gal_data['dec'] <= dec_max) & 
                    (gal_data['phot_g_mean_mag'] <= 22) &
                    gal_data['posangle_sersic'].notna() &
                    gal_data['radius_sersic'].notna() &
                    gal_data['ellipticity_sersic'].notna() &
                    gal_data['n_sersic'].notna() &
                    gal_data['phot_g_mean_mag'].notna() 
                ]
                print(f"select galaxy: {select_data}")

                photons_distribution_simulator.update_galaxy_sn_dict(target_key, select_data)

            if not select_data.empty:
                gal_id = np.array(select_data['source_id'])
                gal_ra = np.array(select_data['ra'])       
                gal_dec = np.array(select_data['dec'])        
                gal_posangle = np.array(select_data['posangle_sersic'])
                gal_e = np.array(select_data['ellipticity_sersic'])
                gal_q = 1 - gal_e 
                gal_n = np.array(select_data['n_sersic'])
                gal_re_mas = np.array(select_data['radius_sersic'])
                gal_re = gal_re_mas / 1000 
                gal_flux = np.array(select_data['phot_g_mean_flux'])
                gal_flux_g = (1050-330) * 1.346109e-21 * gal_flux

                if self.ADR_flag == True:
                    error_ADR_coordinates_gal = self.ADR(self.world.time_astropy.tdb.jd, gal_ra, gal_dec)
                    gal_ra += error_ADR_coordinates_gal[0]
                    gal_dec += error_ADR_coordinates_gal[1]

                gal_cord = np.column_stack((gal_ra, gal_dec))
                gal_pix = self.world.telescope.wcs.wcs_world2pix(gal_cord, 0)

                # 创建掩码，选择在相机范围内的点
                valid_mask = (
                    (gal_pix[:, 0] >= 0) & 
                    (gal_pix[:, 0] < self.world.telescope.camera.pixel_number_x) & 
                    (gal_pix[:, 1] >= 0) & 
                    (gal_pix[:, 1] < self.world.telescope.camera.pixel_number_y)
                )

                # 应用掩码过滤所有相关数组
                select_data = select_data[valid_mask]
                gal_pix = gal_pix[valid_mask]
                gal_id = gal_id[valid_mask]
                gal_ra = gal_ra[valid_mask]
                gal_dec = gal_dec[valid_mask]
                gal_posangle = gal_posangle[valid_mask]
                gal_q = gal_q[valid_mask]
                gal_n = gal_n[valid_mask]
                gal_re = gal_re[valid_mask]
                gal_flux_g = gal_flux_g[valid_mask]
                print("number of galaxies in the field of view:", len(gal_id))
                print("galaxy's pix:", gal_pix)

                error_fraction_jitter_sampl_gal = self.jitter_error(select_data)
                luminosity_fraction_extinction_gal = self.extinction_error(gal_ra, gal_dec)

                gal_flux = np.array(gal_flux_g, dtype=np.float64)
                gal_ADU = gal_flux * error_fraction_jitter_sampl_gal * luminosity_fraction_extinction_gal 
                # * self.world.telescope.camera.QE / self.world.telescope.camera.gain
                # print(f"galaxy's flux and ADU stats: {np.min(gal_flux)}, {np.max(gal_ADU)}")
                for i, (x, y, f) in enumerate(zip(gal_pix[:, 0], gal_pix[:, 1], gal_ADU)):
                    local_size = self.galaxy_local_size
                    height, width = image_gal.shape
                    y_min, y_max = int(y - local_size // 2), int(y + local_size // 2)
                    x_min, x_max = int(x - local_size // 2), int(x + local_size // 2)
                    y_min = max(0, y_min)
                    y_max = min(height, y_max)
                    x_min = max(0, x_min)
                    x_max = min(width, x_max)
                    if y_max<=0 or y_min >= image_gal.shape[0] or x_max<=0 or x_min>= image_gal.shape[1]:
                        continue
                    # profile
                    local_image = galsim.ImageF(local_size, local_size, scale=self.world.telescope.arcsec_pixel_1)
                    print("local_image min/max before adding:", local_image.array.min(), local_image.array.max())
                    gsparams = galsim.GSParams(maximum_fft_size=60000)
                    # print("gsparams min/max:", gsparams.array.min(), gsparams.array.max())
                    n = min(5.2, gal_n[i])
                    re = gal_re[i]
                    gal = galsim.Sersic(n, half_light_radius=re, gsparams=gsparams)
                    gal = gal.withFlux(f)
                    # print("gal min/max at withFlux:", gal.array.min(), gal.array.max())
                    # shape
                    gal_axis_ratio = gal_q[i]
                    gal_beta = gal_posangle[i]
                    gal_shape = galsim.Shear(q=gal_axis_ratio, beta=gal_beta * galsim.degrees)
                    gal = gal.shear(gal_shape)
                    gal.drawImage(local_image) 
                    print("local_image min/max at drawImage:", local_image.array.min(), local_image.array.max())
                    print("n, re, gal_axis_ratio, gal_beta", n, re, gal_axis_ratio, gal_beta)
                    image_gal[y_min:y_max, x_min:x_max] += local_image.array[:y_max-y_min, :x_max-x_min]
                    if np.any(image_gal < 0):
                        print("Warning: galaxy field contains non-positive values. Clipping to 0.")
                        print(f"the total number of non-positive values: {np.sum(image_gal <= 0)}")
                        image_gal = np.clip(image_gal, 0, None)
                    print(f"image_gal min/max after adding:", image_gal.min(), image_gal.max())

            # Consider supernova erupt
            if supernova_erupt:
                def rejection_sampling_sne(gal, R_model, K, seed=None):
                    rng = np.random.RandomState(seed)
                    N = len(gal)
                    z = gal['redshift_ugc'].values
                    R = R_model(z)
                    g = np.ones(N)/N
                    M = N * R.max()
                    # rejection sample
                    S = []
                    count = 0
                    while count < K:
                        j = rng.choice(N, p=g)   #  An element is selected from the indexed set {0,1,...,N-1} and the probability that the ith one is selected is g_i  
                        u = rng.uniform(0, M*g[j])
                        if u < R[j]:
                            S.append(j)
                            count +=1
                    # print(gal, R, K, S)
                    return S
                def R_Ia(z):
                    return 0.3e-3 * (1 + z)**(-1.5)  # unit yr⁻¹ Mpc⁻³

                supernova_erupt_events = pd.read_csv(supernova_erupt_catalog)
                new_gal_sn_data = []
                supernova_erupt_num = supernova_erupt_events['number'].count() 
                gal_supernova = select_data[select_data['redshift_ugc'].notna()]
                if (not gal_supernova.empty) and (supernova_erupt_num != 0):
                    sn_indices = rejection_sampling_sne(gal_supernova, R_Ia, supernova_erupt_num, self.supernova_seed)
                    id_sn_erupt = [gal_supernova['source_id'].values[i] for i in sn_indices]
                    ra_sn_erupt = [gal_supernova['ra'].values[i] for i in sn_indices]
                    dec_sn_erupt = [gal_supernova['dec'].values[i] for i in sn_indices] 
                    z_sn_erupt = [gal_supernova['redshift_ugc'].values[i] for i in sn_indices]   
                    pos_sn_erupt = SkyCoord(ra=ra_sn_erupt*u.deg, dec=dec_sn_erupt*u.deg, frame='icrs',location=self.world.telescope.position)
                    ltt_bary = self.world.time_astropy.light_travel_time(pos_sn_erupt) 
                    t_tdb_jd = self.world.time_astropy.tdb.jd           
                    time_barycentre = t_tdb_jd + ltt_bary 
                    flux_sn_erupt = []
                    # print(gal_supernova, supernova_erupt_num, sn_indices, z_sn_erupt)

                    df = pd.read_csv(self.filter)
                    band_name = self.band_name
                    mag_system = self.mag_system
                    wavelength = np.asarray(df['wavelength']) 
                    transmission = np.asarray(df['throughput_frac'])
                    band = sncosmo.Bandpass(wavelength, transmission, name=band_name)
                    sncosmo.register(band, band_name, force=True)

                    for index,row in supernova_erupt_events.iterrows():
                        model = sncosmo.Model(source='salt3')
                        model.set(z=z_sn_erupt[index])  # redshift
                        t0_sn_erupt = row["t0"]
                        x1_sn_erupt = row["x1"]
                        c_sn_erupt = row["c"]
                        abmag_sn_erupt = row["abmag"]
                        print(f"Supernova erupt parameters: z={z_sn_erupt[index]}, t0={t0_sn_erupt}, x1={x1_sn_erupt}, c={c_sn_erupt}, abmag={abmag_sn_erupt}")
                    # the magnitude corresponding to the flux must be the same with the gaia calculation
                        model.set_source_peakabsmag(abmag_sn_erupt, band_name, mag_system)  # peak absolutly magnitude -5
                        params = {
                            't0': t0_sn_erupt,  
                            'x1': x1_sn_erupt,      
                            'c': c_sn_erupt      
                        }
                        model.set(**params)
                        lc_sn_erupt = model.bandflux(band_name, time_barycentre.jd, zp=self.zero_mag, zpsys=mag_system)  # erg/s/cm^2/A
                        flux_sn_erupt.append(lc_sn_erupt * 1e-3 * self.T_lambda * 1e-10)  # change to J/s/m^2
                        print(f"Supernova erupt flux at t (JD): {lc_sn_erupt}, converted flux: {lc_sn_erupt * 1e-3 * self.T_lambda * 1e-10} J/s/m^2")

                    if self.ADR_flag == True:
                        error_ADR_coordinates_sn = self.ADR(t_tdb_jd, ra_sn_erupt, dec_sn_erupt)
                        ra_sn_erupt += error_ADR_coordinates_sn[0]
                        dec_sn_erupt += error_ADR_coordinates_sn[1]

                    luminosity_fraction_extinction_sn_erupt = self.extinction_error(ra_sn_erupt, dec_sn_erupt)
                    # normalize_factor_sn_erupt =  (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s  * (self.alpha - 1)/ (self.gamma**2 * self.hnu)
                    n_photon = np.array(flux_sn_erupt)  * luminosity_fraction_extinction_sn_erupt
                    # * normalize_factor_sn_erupt
                    print(f"Supernova erupt n_photon: {n_photon}")
                    radec_sn_erupt = np.array([[ra_sn_erupt[i],dec_sn_erupt[i]] for i in range(len(ra_sn_erupt))])
                    pixcrd = self.world.telescope.wcs.wcs_world2pix(radec_sn_erupt, 0)
                    FWHM = self.FWHM
                    moffat_pixel = self.moffat_scale_FWHM * FWHM  

                    x0 = [item[0] for item in pixcrd]
                    y0 = [item[1] for item in pixcrd]
                    for x0, y0, A in zip(x0, y0, n_photon):
                        x_min = max(int(x0 - moffat_pixel), 0)
                        x_max = min(int(x0 + moffat_pixel), img.shape[1])
                        y_min = max(int(y0 - moffat_pixel), 0)
                        y_max = min(int(y0 + moffat_pixel), img.shape[0])
                        if x_min >= x_max or y_min >= y_max:
                            continue  
                        X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
                        lower = self.FWHM*(1-3*self.FWHM_error)
                        upper = self.FWHM*(1+3*self.FWHM_error)
                        FWHM_each = np.clip(np.random.normal(loc=self.FWHM, scale=self.FWHM_error * self.FWHM),
                                 a_min=lower, a_max=upper)
                        gamma_each = FWHM_each / (2 * np.sqrt(2**(1/self.alpha) - 1))
                        normalize_factor_each =  (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s  * (self.alpha - 1)/ (gamma_each**2 * self.hnu)
                        A_each = A * normalize_factor_each
                        g = Moffat2D(amplitude=A_each, x_0=x0, y_0=y0, gamma=gamma_each, alpha=self.alpha)
                        image_supernova[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)

                    for i in range(len(id_sn_erupt)):
                        new_gal_sn_data.append({
                            'Gaia DR3 id': id_sn_erupt[i],
                            'ra': ra_sn_erupt[i],
                            'dec': dec_sn_erupt[i],
                            'pixcrd': pixcrd[i], 
                            'z': z_sn_erupt[i], 
                            'tdb_jd': t_tdb_jd,
                            'tdb_bjd': time_barycentre.jd, 
                            'flux': flux_sn_erupt[i]
                        })

                if new_gal_sn_data:
                    gal_supernova_path = Path(self.world.target_dir) / 'erupt_supernova_inputdata_sorted.csv'
                    new_df = pd.DataFrame(new_gal_sn_data)
                    if gal_supernova_path.exists() and gal_supernova_path.stat().st_size > 0:
                        try:
                            existing_df = pd.read_csv(gal_supernova_path)
                            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        except (pd.errors.EmptyDataError, pd.errors.ParserError):
                            combined_df = new_df
                    else:
                        combined_df = new_df
                    combined_df = combined_df.drop_duplicates(subset=['Gaia DR3 id', 'tdb_jd'], keep='last') ## drop duplicates (based on a combination of target_id and time, keeping the latest flux value)
                    combined_df = combined_df.sort_values(['Gaia DR3 id', 'tdb_jd'])
                    combined_df = combined_df.reset_index(drop=True)
                    combined_df.to_csv(gal_supernova_path, index=False)                

            image_sn_gal = image_supernova +image_gal
            return image_sn_gal
    
    def generate_photon(self):
        img = np.zeros((self.world.telescope.camera.pixel_number_y, self.world.telescope.camera.pixel_number_x))
        sky = self.sky(img)
        if np.any(np.isnan(sky)):
            print("Warning: sky field contains NaN. Replacing with mean value.")
            print(f'the total number of NaN values: {np.sum(np.isnan(sky))}')
            sky = np.nan_to_num(sky, nan=np.mean(sky))
        if np.any(sky < 0):
            print("Warning: sky field contains non-positive values. Clipping to 0.")
            print(f"the total number of non-positive values: {np.sum(sky <= 0)}")
            sky = np.clip(sky, 0, None)
        flux_variable_star = self.flux_variable_all_photons(img)
        if np.any(np.isnan(flux_variable_star)):
            print("Warning: flux_variable_star field contains NaN. Replacing with mean value.")
            print(f'the total number of NaN values: {np.sum(np.isnan(flux_variable_star))}')
            flux_variable_star = np.nan_to_num(flux_variable_star, nan=np.mean(flux_variable_star))
        if np.any(flux_variable_star < 0):
            print("Warning: flux_variable_star field contains non-positive values. Clipping to 0.")
            print(f"the total number of non-positive values: {np.sum(flux_variable_star <= 0)}")
            flux_variable_star = np.clip(flux_variable_star, 0, None)
        satellite = self.satellite(img)
        if np.any(np.isnan(satellite)):
            print("Warning: satellite field contains NaN. Replacing with mean value.")
            print(f'the total number of NaN values: {np.sum(np.isnan(satellite))}')
            satellite = np.nan_to_num(satellite, nan=np.mean(satellite))
        if np.any(satellite < 0):
            print("Warning: satellite field contains non-positive values. Clipping to 0.")
            print(f"the total number of non-positive values: {np.sum(satellite <= 0)}")
            satellite = np.clip(satellite, 0, None)
        asteroid = self.asteroid(img)
        if np.any(np.isnan(asteroid)):
            print("Warning: asteroid field contains NaN. Replacing with mean value.")
            print(f'the total number of NaN values: {np.sum(np.isnan(asteroid))}')
            asteroid = np.nan_to_num(asteroid, nan=np.mean(asteroid))
        if np.any(asteroid < 0):
            print("Warning: asteroid field contains non-positive values. Clipping to 0.")
            print(f"the total number of non-positive values: {np.sum(asteroid <= 0)}")
            asteroid = np.clip(asteroid, 0, None)
        galaxy = self.galaxy_supernova(img)
        if np.any(np.isnan(galaxy)):
            print("Warning: galaxy field contains NaN. Replacing with mean value.")
            print(f'the total number of NaN values: {np.sum(np.isnan(galaxy))}')
            galaxy = np.nan_to_num(galaxy, nan=np.mean(galaxy))
        if np.any(galaxy < 0):
            print("Warning: galaxy field contains non-positive values. Clipping to 0.")
            print(f"the total number of non-positive values: {np.sum(galaxy <= 0)}")
            galaxy = np.clip(galaxy, 0, None)

        scintillation = self.scintillation(img)
        total_photon = (sky + flux_variable_star  + satellite + asteroid + galaxy) * scintillation

        return total_photon


class sensor:
    """high-level CCD/CMOS image sensor noise simulation"""

    def __init__(self, photon_simulator, sensor_config):
        self.photon_simulator = photon_simulator
        self.initpar = sensor_config
        for key, val in sensor_config.items():
            setattr(self, key, val)

    def connect(self, world):
        self.world = world
        
    
    # def photon_shot_noise(self):
    #     """ photon_count: np.ndarray of input photons """
    #     photon_count = self.photon_count
    #     return np.random.poisson(photon_count)
    
    def photo_response_non_uniformity(self):
        """
        Photoresponse non-uniformity (PRNU) - Multiplicative Gaussian noise (fixed mode)
        Args:
            sigma_prnu: PRNU Standard deviation
        """

        sigma_prnu = self.sigma_prnu
        prebias = np.random.RandomState(seed=self.seed_prnu)
        prnu_pattern = prebias.normal(0, sigma_prnu, (self.world.telescope.camera.pixel_number_y, self.world.telescope.camera.pixel_number_x))
        # print(f"PRNU pattern: {prnu_pattern}")
        return prnu_pattern
    
    def dark_current(self):
        """" PA: pixel area [cm2], D_FM: the dark current figure-of-merit at 300K [nA/cm2], E_gap: the band gap energy of the semiconductor [eV]"""
        # Theoretical formula to calculate dark current density
        k_b = 8.617 * 10**(-5)  # Boltzmann's constant [eV/K]
        if self.dark_rate_model:
            dark_rate = self.pixel_area * self.dark_figure_merit * (self.temperature ** 1.5) * \
                       np.exp(-self.energy_gap / (2 * k_b * self.temperature))
        else:
            # Use a fixed dark current rate for simplicity
            dark_rate = self.dark_current_rate
      
        dark_signal = self.world.telescope.camera.exposure_s * dark_rate
        # print(f"Dark current signal: {dark_signal}")
        return dark_signal
    
    # def dark_current_shot_noise(self, dark_current_mean):
    #     return np.random.poisson(dark_current_mean, (self.pixel_number_y, self.pixel_number_x))
    
    def dark_current_fpn(self, dark_current: np.ndarray):
        dark_noise_factor = self.dark_noise_factor
        sigma_dc_fpn = dark_current * dark_noise_factor
        # print(f"Dark current FPN standard deviation: {sigma_dc_fpn}")
        prebias = np.random.RandomState(seed=self.seed_dark_fpn)
        dark_fpn_pattern = prebias.lognormal(0, sigma_dc_fpn**2, 
                                                       (self.world.telescope.camera.pixel_number_y, self.world.telescope.camera.pixel_number_x))
        # print(f"Dark current FPN pattern: {dark_fpn_pattern}")
        return dark_fpn_pattern
    
    def source_follower_noise(self, tau_D_default=True):
        """
            f_clock: readout clock frequency (typically several MHz)
            W: thermal white noise in V /Hz**(1/2)
            f_c:flicker noise corner frequency in [Hz]
            tau_RTN: Random Telegraph Noise (RTN) characteristic time constant [sec]
            delta_I: source follower current modulation induced by RTN [A]
            t_s: CDS sample-to-sampling time [sec]
            A_SN: sense node conversion gain (V/e-)
            A_SF: source follower gain(V/V)
        """
        if tau_D_default:
            tau_D = 0.5 * self.t_s
        else:
            tau_D = self.tau_D
        
        # Logarithmic spacing for computational efficiency
        # f_max = min(f_clock, 1e8)  # Limit maximum frequency to avoid computational overload
        f = np.logspace(1, np.log10(self.f_clock), 10000) 
        
        # 1 S_SF(f)
        # white noise
        S_white = self.W**2 * np.ones_like(f)
        # 1/f noise
        S_1f = self.W**2 * self.f_c / f
        # RTN(Random Telegraph Noise)
        S_RTN = (2 * self.delta_I**2 * self.tau_RTN) / (4 + (2 * np.pi * f * self.tau_RTN)**2)
        
        S_SF = S_white + S_1f + S_RTN
        
        # 2. transfer function H_CDS(f)
        # Low-pass filter section
        H_LP = 1 / (1 + (2 * np.pi * f * tau_D)**2)
        # CDS (correlated dual sampling)
        H_CDS_core = 2 - 2 * np.cos(2 * np.pi * f * self.t_s)
      
        H_CDS = H_LP * H_CDS_core
        
        # 3. noise power integral
        df = np.diff(f)
        df = np.append(df, df[-1])  # Last interval
        
        
        sigma_SF_V = np.sqrt(np.sum(S_SF * H_CDS * df))
        conversion_factor = self.A_SN * self.A_SF * (1 - np.exp(-self.t_s / tau_D))
        # print(f"Conversion factor: {conversion_factor}")
        sigma_SF = sigma_SF_V / conversion_factor
        # print(f"Source follower noise standard deviation: {sigma_SF}")
        return np.round(np.random.normal(0, sigma_SF, (self.world.telescope.camera.pixel_number_y, self.world.telescope.camera.pixel_number_x)))
    
    def sensing_node_reset_noise(self):
        k_b = constants.k  # Boltzmann's constant [J/K]
        temperature = self.temperature
        sensing_capacitance = self.sensing_capacitance  # sense node capacitance [F], F=Q/V,
        sigma_reset = np.sqrt(k_b * temperature / sensing_capacitance)
        # print(f"Sensing node reset noise standard deviation: {sigma_reset}")
        return np.random.lognormal(0, sigma_reset**2,(self.world.telescope.camera.pixel_number_y, self.world.telescope.camera.pixel_number_x))
    
    def offset_fpn(self):
        """
        Offset Fixed Mode Noise - Autoregressive Column Correlation Noise
        correlation_factor: a ∈ [0, 0.5]
        U (j) are zero mean, uncorrelated random variables with the variance sigman_U
        """
        n = self.world.telescope.camera.pixel_number_x
        a = self.offset_correlation_factor 
        sigma_U = self.offset_fpn_sigma_U
        prebias = np.random.RandomState(seed=self.seed_offset_fpn)
        U = prebias.normal(0, sigma_U, (n))

        # Creates a compact storage format for banded matrices (3 rows and n columns).
        # Formatting notes:
        # row0: upper diagonal element (the first position not used is complemented by 0)
        # row1: main diagonal element
        # row2: lower diagonal element (last position not used with a zero)
        banded_matrix = np.zeros((3, n))  
        banded_matrix[1, :] = 1  # The main antagonists are all 1
        banded_matrix[0, 1:] = -a  # The upper diagonals are all -a
        banded_matrix[2, :-1] = -a  # The lower diagonals are all -a

        Y = solve_banded((1, 1), banded_matrix, U) # n cloumns
        print(f"Offset FPN pattern: {Y}")
        return Y
    
    def voltage_electron_nonlinearity(self, electron_count):
        """
        V/e⁻ Nonlinear - Exponential Decay Model
        electron_count: array
        V_ref: Reference voltage (V)
        alpha: Nonlinear coefficient
        """
        k_1 = 10.909 * 10**15
        q = constants.elementary_charge
        # print(f"V/e⁻ Nonlinearity: {self.V_ref * np.exp(-self.alpha * electron_count * q / k_1)}")
        return self.V_ref * np.exp(-self.alpha * electron_count * q / k_1)
    
    def voltage_voltage_nonlinearity(self, voltage: np.ndarray):
        """
        V/V nonlinearity - gain varies with signal
        voltage: input voltage
        gamma_nlr: nonlinear ratio
        v_full_well: full well voltage
        """
        gain_addition = (self.gamma_nlr - 1) * voltage / self.v_full_well
        # print(f"V/V Nonlinearity gain addition: {gain_addition}")
        return gain_addition
    
    def A_adc_nonlinearity(self):
        """
        ADC Nonlinearity - Gain Signal Dependence
        gamma_adc_nonlin: ADC nonlinear ratio
        V_adc_ref: ADC reference voltage
        A_adc_linear: ADC linear gain
        """
        alpha_adc = (np.log10(self.gamma_adc_nonlin * self.A_adc_linear)/np.log10(self.A_adc_linear) - 1) / self.V_adc_ref
        # print(f"ADC Nonlinearity gain: {self.A_adc_linear**(1-alpha_adc)}")
        return self.A_adc_linear**(1-alpha_adc)
    
    ## CDS correct simulation and quantization noise
    
    # Simulate vignetting effect: The number of photons arriving at different pixels changes
    def simulate_vignetting(self):
        """
        Parameters:
        -----------
        pixel_number_x/y
        pixel_size: m/pixel
        focal_length: m
        cone_height : H(m)
        self.entrance_radius: radius of incidence circle R(m)
        self.exit_radius: Radius of injection circle r(m)
        lens_radius: l(m) 
        """
        # Image center coordinates
        center_x = self.world.telescope.camera.pixel_number_x / 2
        center_y = self.world.telescope.camera.pixel_number_y / 2  # 前半部分会少一个（e.g. 4的一半是2）
        x = np.arange(self.world.telescope.camera.pixel_number_x)  # 刚刚好从0开始，补上少的1
        y = np.arange(self.world.telescope.camera.pixel_number_y)
        X, Y = np.meshgrid(x, y)
        # 1. the angle of incidence θ
        dx = X - center_x
        dy = Y - center_y
        radial_distance = np.sqrt(dx**2 + dy**2) * self.world.telescope.camera.pixel_size_m
        theta = np.arctan(radial_distance / self.world.telescope.focal_length_m)
        # 2. displacement of projection
        h = self.cone_height * np.tan(theta)
        # Avoid dividing by zero
        h_safe = np.where(h == 0, 1e-10, h)
        # 3.The overlap angles α(θ) and β(θ)
        cos_alpha = (self.entrance_radius**2 - self.exit_radius**2 + h**2) / (2 * self.entrance_radius * h_safe)
        cos_alpha = np.clip(cos_alpha, -1, 1)  # Restricted to the interval [-1, 1]
        alpha = np.arccos(cos_alpha)
        alpha = np.where(h == 0, 0, alpha)
        cos_beta = (self.exit_radius**2 - self.entrance_radius**2 + h**2) / (2 * self.exit_radius * h_safe)
        cos_beta = np.clip(cos_beta, -1, 1)
        beta = np.arccos(cos_beta)
        beta = np.where(h == 0, 0, beta)
        # 4. Effective optical transmission area 
        effective_area =  self.entrance_radius**2 * (alpha - np.sin(alpha) * np.cos(alpha)) + self.exit_radius**2 * (beta - np.sin(beta) * np.cos(beta))
        # print("Effective Area:", effective_area)
        # 5. calculate the relative intensity decay factor B(θ)
        max_area = np.pi * (self.world.telescope.diameter_m/2)**2
        # print("Max Area:", max_area)
        intensity_map = effective_area / max_area
        ny, nx = intensity_map.shape
        cy = ny // 2
        cx = nx // 2
        intensity_map[cy, cx] = 1.0
        print(intensity_map.shape)
        return intensity_map

    def simulate_full_chain(self, frame_type='star', flat_level=None):
        if 'bias' in frame_type:
            self.photon_count = np.zeros((self.world.telescope.camera.pixel_number_y, self.world.telescope.camera.pixel_number_x))
            photons_with_shot = np.random.poisson(self.photon_count)
            photon_count = photons_with_shot * int(self.vignetting_flag) * self.simulate_vignetting() + photons_with_shot * int(not self.vignetting_flag)  # vignetting effect
        elif 'flat' in frame_type:
            self.photon_count = np.full((self.world.telescope.camera.pixel_number_y, self.world.telescope.camera.pixel_number_x), flat_level)
            photons_with_shot = np.random.poisson(self.photon_count)
            photon_count = photons_with_shot * int(self.vignetting_flag) * self.simulate_vignetting() + photons_with_shot * int(not self.vignetting_flag)  # vignetting effect
        else:
            self.photon_count = self.photon_simulator.generate_photon()
            # photon shot noise
            max_photons = np.max(self.photon_count)
            print(f"max photons number: {max_photons:.2e}")
            print(np.mean(self.photon_count))
            print(self.photon_count)
            photons_with_shot = np.random.poisson(self.photon_count)
            # print(f'input photon numbers: {self.photon_count}')
            # print(f'mean input photon numbers: {np.mean(self.photon_count)}')
            photon_count = photons_with_shot * int(self.vignetting_flag) * self.simulate_vignetting() + photons_with_shot * int(not self.vignetting_flag)  # vignetting effect

        print("CCD/CMOS sensor noise simulate...")
        # 1. photons to electrons stage
        print("1. photons to electrons stage...") 

        # transform photons to electrons
        electrons = photon_count * self.world.telescope.camera.QE
    
        # PRNU
        electrons_prnu = electrons*(1 + int(self.PRNU_flag)*self.photo_response_non_uniformity())
        
        # dark current
        dark_current = self.dark_current()
        dark_with_shot = np.random.poisson(dark_current)
        dark_fpn = dark_with_shot*(1 + int(self.dark_current_fpn_flag)*self.dark_current_fpn(dark_current))

        # total electrons
        total_electrons = np.round(electrons_prnu + dark_fpn + int(self.SF_flag)*self.source_follower_noise())
        total_electrons = np.minimum(self.world.telescope.camera.full_well_capacity_ke,total_electrons)  ## limit to full well capacity
        
        
        # 2. charge to voltage stage
        print("2. charge to valtage stage...")
        
        # sense node
        reset_noise = int(self.reset_noise_flag) * self.sensing_node_reset_noise()
        V_ref = self.V_ref + reset_noise
        I_sn_v = int(not self.V_e_nonlinear)*(V_ref - total_electrons*self.A_SN) + int(self.V_e_nonlinear)*self.voltage_electron_nonlinearity(total_electrons)
        # print(f"I_sn_v: {I_sn_v}")

        # source follower
        I_sf_v = I_sn_v * (self.A_SF + int(self.V_v_nonlinear)*self.voltage_voltage_nonlinearity(I_sn_v))

        offset_arr = self.offset_fpn()
        for i in range(self.world.telescope.camera.pixel_number_x):
            sf_noise = np.random.normal(loc=offset_arr[i], scale=abs(0.1*offset_arr[i]), size=self.world.telescope.camera.pixel_number_y)
            I_sf_v[:, i] += int(self.offset_fpn_flag) * sf_noise
        # print(f"I_sf_v: {I_sf_v}")

        # 3. voltage to digital stage
        print("3. voltage to digital stage...")
        
        # ADC nonlinearity
        I_DN =  np.round((int(not self.A_adc_nonlinearity_flag) * self.A_adc_linear + int(self.A_adc_nonlinearity_flag) * self.A_adc_nonlinearity()) * (self.V_adc_ref - I_sf_v))
        I_DN = np.minimum(2**self.world.telescope.camera.bit_per_pixel-1, I_DN) # limit the ADU full well capacity
        # print(f"I_DN: {I_DN}")
        return I_DN


if __name__ == '__main__':


    with open('/Users/kexin_li/Documents/vs_py/sim_events/config_tianyu.json', 'r', encoding='utf-8') as f:
        config = json.load(f)


    camera_par = config['camera_parameters']
    mount_par = config['mount_parameters']
    telescope_par = config['telescope_parameters'].copy()
    
    
    telescope_par['latlonalt'] = tuple(telescope_par['latlonalt'])
    
    photons_config = config['photons_config']
    sensor_config = config['sensor_config']
    
    
    world_config = config['world_config']
    output_path = world_config['output_path']
    julian_date = world_config['julian_date']
    schedule_file = world_config['schedule_file']
    

    cam = camera(camera_par)
    mnt = mount(mount_par)
    tel = telescope(mnt, cam, telescope_par)

    photons = photons_distribution_simulator(photons_config)
    sim = sensor(photons, sensor_config)
    wd = world(tel, photons, sim, output_path, julian_date, input_schedule=schedule_file)
    

    wd.run_sim()




