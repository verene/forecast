
import datetime as dt
from dateutil.relativedelta import relativedelta
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz
from sklearn import metrics
import yaml

#User defined modules
from odw import call_odw_df
from hydro_ops import get_nhmm_natflow, get_usace
from error_email import error_email

#Basic Steps Performed to Validate a Forecast:
# 1. Get observed data
# 2. Get forecast data
# 3. Make climatology (seasonal forecasts) or persistence (short-term forecasts) as baseline forecasts
# 4. For each source, split forecasts into different horizons
# 5. For each source and horizon, align forecasts with observations
# 6. Calculate deterministic stats (raw and skill scores)
# 7. Plot deterministic performance stats by horizon
# 8. Calculate probabilistic performance stats (CRPS and CRPSS) #TODO
# 9. Plot Q-Q and stats by horizon #TODO

#Requires pythonlib repo containing odw, hydro_ops, error_email modules to be located in the same directory, or in the user's PATH.
# Also requires access to SCL Oracle data warehouse where historical forecasts are archived. 

class Verification:
    def __init__(self, sitename, horizontype, sd, ed, inc='M', obsthresh=0.7, snow_survey=False, bymonth=False, localtz='US/Pacific'):
        '''
        A verification object is a collection of streamflow forecast objects, observations, and metadata for
        a single streamflow forecast site.
        Once forecast objects, and observations have been added to the verification, a baseline climatology
        forecast can be calculated (based on historical observations). The skill scores of each forecast
        can then be calculated and plotted by horizon for comparison.
        Parameters:
        - sitename: Name of site. Required, no default. Must be defined in both forecast_metadata.yaml and obs_metadata.yaml
        - horizontype: Type of forecast; short-term or seasonal. Required, no default.
        - sd: Desired start date of forecast evalution period (datetime object). Required, no default.
        - ed: Desired end date of forecast evaluation period (datetime object). Required, no default.
        - inc: Forecast interval period; M=monthly (default), D=daily, H=hourly, Q=quarterly.
        - obsthresh: Fractional threshold; if aggregated data have counts less than this fraction, the
           data point is dropped from the evaluation. Default value = 0.7
        - snow_survey: True or False (default) flag indicating whether the snow survey forecast has been added
           to the verification. If it has, the verification is only performed at a monthly interval on
           valid forecast calendar months of the year (January through September only)
        - bymonth: True or False (default) flag; for a monthly forecast interval, evaluate forecast performance
           on each calendar month of the year separately
        '''
        self.forecasts = []
        self.obsdata = None
        self.climodata = None
        self.site = sitename
        self.sd = sd #Requested start date
        self.ed = ed #Requested end date
        self.inc = inc #Time increment for forecast performance evaluation
        self.horizontype = horizontype #Type of forecast: short-term or seasonal
        self.drop_lt = obsthresh #Fractional threshold for dropping aggregated data
        self.fxstart = None #Actual min forecast start time
        self.fxend = None #Actual max forecast end time
        self.snow_survey = snow_survey #Only evaluate on the reduced set of months included in the snow surveys
        self.bymonth = bymonth #Perfom verification for each horizon on each month of the year separately
        self.localtz = localtz
    
    def get_forecast_sources(self):
        '''Prints a list of forecast sources that have been added to the verification object.'''
        return [self.forecasts[i].source for i in range(len(self.foreasts))]

    def add_forecast(self, source, localtz='US/Pacific'):
        '''
        Adds a forecast object to the verification object.
        If snow survey forecast is being added, sets snow_survey attribute to True.
        Updates the verification object's start date to equal the forecast object's start date if
        no start date exists or if the new forecast object's start date is earlier than existing vx start date.
        Updates the verification object's end date to equal the forecast object's end date if no end
        date exists or if the new forecast object's end date is later than the existing vx end date.
        '''
        #Check for snow survey source
        if 'snow survey' == source.lower():
            self.snow_survey = True
        self.forecasts += [Forecast(source, self.horizontype, self.sd, self.ed, self.site, self.inc, self.bymonth, localtz=localtz)]

        #Save (or update) the earliest forecast start date in the verification
        try:
            if self.fxstart is not None:
                if self.forecasts[-1].fxdata.validtime.iloc[0] < self.fxstart:
                    self.fxstart = self.forecasts[-1].fxdata.validtime.iloc[0]
            else:
                self.fxstart = self.forecasts[-1].fxdata.validtime.iloc[0]
        except Exception as e:
            print(e)
            pass

        #Save (or update) the latest forecast end date in the verification
        try:
            if self.fxend is not None:
                if self.forecasts[-1].fxdata.validtime.iloc[-1] > self.fxend:
                    self.fxend = self.forecasts[-1].fxdata.validtime.iloc[-1]
            else:
                self.fxend = self.forecasts[-1].fxdata.validtime.iloc[-1]
        except Exception as e:
            print(e)
            pass

        return

    def add_climo(self, climo_sd=None, climo_ed=None, extended_hist=False, localtz='US/Pacific'):
        '''
        Adds a baseline forecast based on climatology to the verificaiton object.
        Construction of the forecast happens in the function make_climo_fx().
        '''
        print("**Adding climo forecast for {}. Please do not add any more forecasts to this verification."\
            "".format(self.site))
    
        #Find the max forecast horizon among forecasts added to the verification object.
        minmax_h = np.inf
        for fxo in self.forecasts:
            self.max_h = min(minmax_h,max(fxo.horizons))

        #Generate the climo forecast; save returned forecast data and horizons
        cfx, hs = make_climo_fx(self, climo_sd, climo_ed, extended_hist=extended_hist, localtz=localtz)
        #Create a forecast object for the climo forecast
        self.climo = Forecast(fxsource="Climatology", horizontype='deterministic', sd=cfx.index[0],
                            ed=cfx.index[-1], site=self.site, inc=self.inc, bymonth=self.bymonth, localtz=localtz)
        self.climo.fxdata = cfx
        self.climo.horizons = sorted(hs)

        #Calculate the (raw) climo forecast performance metrics
        calc_fx_stats(self, calc_climo=True, norm=False)
        return

    def add_obs(self, drop_lt=None):
        '''
        Add observed streamflow data for the site to the verification object. 
        '''
        totalcounts = { '15mH': 4, '15mD': 4*24, '15mM': 4*24*30, '15mQ': 4*24*30*3,
            'HH': 1, 'HD': 24, 'HM': 24*30, 'HQ':24*30*3,
            'DD': 1, 'DM': 30, 'DQ': 30*3,
            'MM': 1, 'MQ': 3
        }
        print("**Adding obs for {}".format(self.site))

        #Retrieve observed data for the forecast site and add it to the verification object.
        obs, samp = get_obs_wrapper(self.sd, self.ed, sitename=self.site)
        #Resample the obs data from their native temporal resolution to the resolution of the
        # verification object, dropping any intervals that have an insufficient data counts.
        try:
            drop_lt = self.drop_lt*totalcounts[samp+self.inc]
            assert(isinstance(drop_lt,float))
        except:
            print("Unrecognized native obs data resolution.")
            raise
        obs_count = obs.resample(self.inc, closed='right', label='left').count()
        obs = obs.resample(self.inc, closed='right', label='left').mean()
        obs = obs[obs_count > drop_lt].copy()
        if obs.isna().values.all():
            print("ERROR: No observation data remain after dropping records with counts less than the threshold.")
            raise
        obs.index = [dttm + dt.timedelta(days=1) for dttm in obs.index]
        obs.index.names = ['DateTime']
        obs.rename({list(obs.columns)[0]:'Obs'}, axis=1, inplace=True)
        self.obsdata = obs
        return
        
class Forecast:
    def __init__(self, fxsource, horizontype, sd, ed, site, inc, bymonth, localtz='US/Pacific'):
        '''
        Forecast object definition. Forecasts for a given site and from a specified source (e.g. RFC,
        private vendor) may be added to the forecast verification object for that site.
        Parameters:
        - fxsource: Forecast source name. Currently supported: rfc, upstream, vendor1, snow survey
        - horizontype: Type of forecast; short-term or seasonal.
        - sd: Requested start date of forecast valid times (datetime object)
        - ed: Requested end date of forecast valid times (datetime object)
        - site: Site name. In order to be supported, site data (including name) must be defined in
           forecast_metadata.yaml
        - inc: Forecast interval. Currently supported: M=monthly, D=daily, H=hourly, Q=quarterly
        - bymonth: Evaluate forecast performance separately for each calendar month of the evaluation period
        '''
        self.source = fxsource
        self.horizontype = horizontype
        self.sd = sd #Requested start date (actual is determined by fxdata index)
        self.ed = ed #Requested end date (actual is determined by fxdata index)
        self.inc = inc
        self.bymonth = bymonth
        self.stats = {}

        #Unless this is the climatology forecast being instantiated, retrieve parameters from
        # yaml file for retrieving archived forecasts from the database.
        if fxsource !='Climatology':
            self.stats['norm'] = {}
            auto_path = "O:\\POOL\\PRIVATE\\Power Management\\ResAdmin\\automated_reports\\verification\\"
            meta_fn = "forecast_metadata.yaml"
            with open(auto_path+meta_fn) as mf:
                (d)=yaml.full_load(mf)
            metadict = d[horizontype][fxsource]
            self.db = metadict['db']
            self.table = metadict['table']
            self.db_id = metadict['sites'][site]
            self.db_sites = metadict['sites']
            self.createname = metadict['create']
            self.validname = metadict['valid']
            self.sitename = metadict['site'] #DB site column name
            self.deterministicname = metadict['deterministic']
            if 'sidestream' in metadict.keys():
                if metadict['sidestream']:
                    self.target = 'sidestream'
                else:
                    self.target = 'natural'
            else:
                self.target = 'natural'

            #TODO: make this work for pvals != None
            #TODO: make this work if the forecast is SS (query site and all upstream sites
            # and sum pivot over retrieved sites)
            if site is not None:
                if site not in self.db_sites:
                    msg = "WARNING: no forecast retrieved. Site given was {} but must be one of: {}"\
                        "".format(site, ", ".join(self.sites))
                    print(msg)
                else:
                    self.fxdata = get_fx(sitename=self.db_id, fxsource=fxsource, horizontype=horizontype,
                        fx_type='deterministic', sd=self.sd, ed=self.ed, localtz=localtz, inc=self.inc)
                    calc_horizons(self)
            else:
                print("WARNING: Use add_forecast for one of the available sites: {}".format(", ".join(self.sites)))
        else:
            self.stats['raw'] = {}

    def get_target(self):
        return self.target

    def get_fx_data(self):
        return self.fxdata
    

def get_scl_obs(db, table, sitename, sitecode, sd=None, ed=None, lastx=None, dtname='archtime', localtz='US/Pacific', converttz=None):
    '''
    Retrieves obs stored in the Oracle Data Warehouse at native resolution. Inner joins with forecasts and then resample, to
    ensure same counts.
    Parameters:
    - db: Database schema name. Required, no default
    - table: Database table name. Required, no default
    - sitename: Name of site for which to retrieve observed data. Required, no default.
    - sitecode: Database site code used to query observed data. Required, no default.
    - sd: Start date for verification. Default: None. If None specified, and time interval for validation
       (lastx) is not specified, set equal to 30 days ago from current date.
    - ed: End date for verification. Default: None. If None specified, set equal to the current date.
    - lastx: Most recent number of days (relative to end date to perform verification). Default: None.
       If none specified, set equal to 30 days.
    - dtname: Column name in the database in which date / datetime is stored.
    - localtz: Time zone local to the site. Most sites' data used at SCL are in Pacific Time (default).
    - converttz: Time zone to convert to, if site's local time (localtz) is something other than the
       time zone to be evalutated. Default: None.
    '''

    if ed==None:
        ed = dt.datetime.now()
    if sd==None:
        if lastx==None:
            lastx=30
        sd = ed-dt.timedelta(days=lastx)
    obs_q = "select {dttm}, value from {db}.{tb} where to_char({dttm}, 'YYYY-MM-DD') between '{sd}' "\
        "and '{ed}' and b1='{b1}' and elem = '{elem}' order by {dttm}"\
        "".format(dttm=dtname, db=db, tb=table, sd=sd.strftime("%Y-%m-%d"), ed=ed.strftime("%Y-%m-%d"),
            b1=sitecode, elem='Nat Flow')

    obs = call_odw_df(obs_q, index_col=dtname.upper(), parse_dates=True)
    obs.rename({'VALUE': sitename}, axis=1, inplace=True)
    #make obs tz-aware if they aren't already
    try:
        obs.index = obs.index.tz_localize(localtz)
        if localtz != converttz and converttz is not None:
            obs.index = obs.index.tz_convert(converttz)
    except:
        #print("Obs are already tz aware")
        pass
    return obs

def get_fx(sitename, fxsource='rfc', horizontype='seasonal', fx_type='deterministic', pvals=None,
    sd=None, ed=None, lastx=None, localtz='US/Pacific', converttz=None, inc='M'):
    '''
    Returns requested historical forecast data stored in the Oracle Data Warehouse, ordered by
      createtime and validtime.
    The parameters for the ODW query depend on the values passed and are retrieved from the
      forecast_metadata.yaml
    Parameters:
    - sitename: Name of site. Site metadata must be defined in forecast_metadata.yaml in order to be
       supported. (Required; no default)
    - fxsource: indicates the original source of the forecast. Currently supported: rfc (default),
       upstream, vendor1, snow survey
    - horizontype: type of forecast. Currently supported: short-term, seasonal (default)
    - fx_type: deterministic (default) or probabilistic; any value starting with a D or d is interpreted
       as 'deterministic' and any value starting with a P or p is interpreted as probabilisitc.
    - pvals: Requested prediction interval values to validate if fx_type='probabilistic'; must be None,
       or a number between 0 and 100, exclusive. Default: None.
    - sd: Start date for verification. Default: None. If None specified, and time interval for validation
       (lastx) is not specified, set equal to 30 days ago from current date.
    - ed: End date for verification. Default: None. If None specified, set equal to the current date.
    - lastx: Most recent number of days (relative to end date to perform verification). Default: None.
       If none specified, set equal to 30 days.
    - localtz: Time zone local to the site. Most sites' data used at SCL are in Pacific Time (default).
    - converttz: Time zone to convert to, if site's local time (localtz) is something other than the
       time zone to be evalutated. Default: None.
    - inc: Time increment for forecast evaluation. Default: 'M' (monthly)
    '''

    if ed==None:
        ed = dt.datetime.now()
    if sd==None:
        if lastx==None:
            lastx=30
        sd = ed-dt.timedelta(days=lastx)

    #Retrieve forecast metadata for querying the database for archived forecast. get_fx_meta reads metadata
    # from forecast_metadata.yaml
    fx_meta = get_fx_meta(fxsource, horizontype)
    if fx_type[0].lower()=='d':
        ftype = 'deterministic'
    elif fx_type[0].lower()=='p':
        ftype = 'pvals'
    try:
        fxlist = f"avg({fx_meta[ftype]})" #list of columns to select in sql query
    except Exception as e:
        errmsg = 'Error! Forecast type {} is not available for {} {} forecasts.\n'\
            'Options are: {}'.format(ftype, fxsource, horizontype, list(fx_meta.keys()))
        print(errmsg)
        error_email(errmsg=errmsg, e=e, origin_script=__name__)
        raise
    
    #If forecast is probabilistic, figure out which columns to select
    if ftype == 'pvals':
        if pvals != None:
            if isinstance(pvals, str):
                pvals=[pvals]
            else:
                assert isinstance(pvals, list), "ERROR: pvals must be a 'None', string, or list."
        else:
            pvals = list(fxlist.keys())
        cols = [fx_meta[ftype][pvals[i]] for i in range(len(pvals))]
        fxlist = "avg("+"), avg(".join(cols)+")"
        if not len(fxlist):
            raise("ERROR: no forecast columns to select.")

    #Query database
    print("**Getting forecast data for {} from {}".format(sitename, fxsource.capitalize()))
    charstr = {'Q':'YYYY-Q', 'M': 'YYYY-MM', 'D': 'YYYY-MM-DD', 'H': 'YYYY-MM-DD HH24'}
    tstr = {'Q': ' 00:00:00', 'M': '-01 00:00:00', 'D': ' 00:00:00', 'H': ':00:00'}
    csd = sd-relativedelta(months=13)

    fx_q = "select to_char({cttm}, '{charstr}') as ct, {cttm}, to_char({dttm}, '{charstr}')||'{tstr}' as vt, {fxlist} as afv from {db}.{table} "\
        "where to_char({dttm}, 'YYYY-MM-DD') between '{sd}' and '{ed}' and {siteid} = '{stnm}' and to_char({cttm}, 'YYYY-MM') < to_char({dttm}, 'YYYY-MM') and "\
        "to_char({cttm}, 'YYYY-MM') >= '{csd}' and {fxval} <= 200000 group by {cttm}, to_char({dttm}, '{charstr}')||'{tstr}' order by {cttm}, vt"\
        "".format(dttm=fx_meta['valid'], cttm=fx_meta['create'], charstr=charstr[inc], tstr=tstr[inc], db=fx_meta['db'],
            table=fx_meta['table'], sd=sd.strftime("%Y-%m-%d"), fxlist=fxlist, ed=ed.strftime("%Y-%m-%d"), stnm=sitename,
            siteid=fx_meta['site'], csd=csd.strftime("%Y-%m"), fxval=fx_meta[ftype])
    print(fx_q)
    fx = call_odw_df(fx_q, sn='pwmdwd', perm='readwrite', parse_dates=True)

    fx.rename({fx_meta['create'].upper():'createtime', 'VT': 'validtime'}, axis=1, inplace=True)
    fx.validtime = pd.to_datetime(fx.validtime).dt.tz_localize(localtz)
    try:
        fx.createtime = fx.createtime.dt.tz_localize(localtz)
    except Exception as e:
        print(f"WARNING: could not localize createtime\n{e}")
    fx.drop_duplicates(subset=['validtime', 'CT'], keep='first', inplace=True, ignore_index=True)
    fx = fx[['createtime', 'validtime', 'AFV']]
    if fx_type=='deterministic':
        fx.rename({'AFV': fxsource+" "+fx_type}, axis=1, inplace=True)
    #TODO: else... rename pval columns to something useful

    #Convert timezones if needed
    if localtz!=converttz and converttz is not None:
        for f in [fx.validtime, fx.createttime]:
            f.dt.tz_convert(converttz)

    #Return forecast data
    return fx

def calc_horizons(fxo):
    '''
    For the given forecast object, finds a sorted list of forecast horizons
    and stores the list of horizons in the object's "horizons" attribute.
    Horizons are calculated in the temporal units of the forecast object.
    Parameter: fxo: forecast object.
    '''

    fxd = fxo.get_fx_data()
    #Create a list of timedelta objects based on valid time and create time of each forecast point
    r = [relativedelta(fxd['validtime'][i],fxd['createtime'][i]+dt.timedelta(seconds=1)) for i in range(len(fxd))]

    #Calculate the horizon (integer) of each forecast point, depending on the forecast temporal resolution
    if fxo.inc == 'M':
        fxd["horizon"] = [r[i].months+12*r[i].years+1 for i in range(len(r))]
    elif fxo.inc == 'H':
        fxd["horizon"] = [r[i].seconds//3600+1 for i in range(len(r))]
    elif fxo.inc == 'D':
        fxd["horizon"] = [r[i].days+365.25*r[i].years+1 for i in range(len(r))]
    elif fxo.inc == 'Q':
        fxd["horizon"] = [r[i].months//3+4*r[i].years+1 for i in range(len(r))]
    
    #Get unique values and store them in the forecast object's "horizons" attribute.
    hzs = fxd.horizon.unique().tolist()
    fxo.horizons = sorted(hzs)
    return

def make_climo_fx(vx, climo_sd=None, climo_ed=None, extended_hist=False, localtz='US/Pacific'):
    '''
    Creates a baseline forecast based on climatology (average historical observations with the same
    granularity as the forecasts) to use in the calculation of the error metric skill scores.
    For fairness, the period of observed data used to calculate the climatology forecast entends up
    until the start of the forecasts; this way know future data are incorporated into the baseline
    forecasts as they would have been made at the time.
    Paremeters:
    - vx: Verification object (required)
    - climo_sd: Start date for climatology baseline forecast. Default: None. If None specified, the
       start date should be the start of the period of record of the Oracle Data Warehouse (1997-01-01)
    - climo_ed: End date for climatology baseline forecast. Default: None. If None specified, the end
       date should be set to just before the start date of the earliest forecast in the verification object.
    - extended_hist: Retrieve extended historical observations for Ross, Newhalem, or Marblemount
       not available from Oracle Data Warehouse, but stored in a csv on the O drive. Options: True
       or False (default)
    '''
    snow_survey=vx.snow_survey
    #Not all forecasts and horizons should be verified if being compared to the snow survey forecasts.

    #Get recent history from Oracle
    if climo_ed==None:
        climo_ed = vx.fxstart-dt.timedelta(hours=1)
    if not climo_ed.tzinfo:
        climo_ed=pytz.timezone(localtz).localize(climo_ed)
    if climo_sd==None:
        climo_sd = pytz.timezone(localtz).localize(dt.datetime(1997,1,1,0,0,0))
    elif not climo_sd.tzinfo:
        climo_sd=pytz.timezone(localtz).localize(climo_sd)
    climo, samp = get_obs_wrapper(climo_sd, climo_ed, sitename=vx.site)
    climo.rename({vx.site: "Climo"}, inplace=True, axis=1)

    #Optionally, get extended history back to 1982 (or 1989 for Newhalem and Marblemount) from LRDB csv.
    if extended_hist and vx.inc!='H':
        readdir = "O:\\POOL\\PRIVATE\\Power Managment\\ResAdmin\\automated_reports\\skagit\\"
        readfn = "Skagit_natural_inflows_1982-1996.csv"
        usecols = ['Date', '{}'.format(vx.site)]
        ext_dat = pd.read_csv(readdir+readfn, skiprows=1, index_col=0, header=0, use_cols=usecols, parse_dates=True)
        ext_dat.dropna(axis=0, how='any', inplace=True)
        ext_dat.rename({vx.site: "Climo"}, inplace=True, axis=1)
        climo = pd.concat([ext_dat,climo], axis=0)

    if vx.inc=='M':
        climo_agg = climo.groupby(climo.index.month).mean()
    elif vx.inc =='D':
        climo_agg = climo.groupby(climo.index.day).mean()
    elif vx.inc =='Q':
        climo_agg = climo.groupby(climo.index.quarter).mean()
    if vx.fxend and vx.fxstart:
        climo_ed_ext=dt.datetime(vx.fxend.year+1, 1, 1, 0, 0, 0)
        climo_sd_ext=dt.datetime(vx.fxstart.year, 1, 1, 0, 0, 0)
    else:
        print("ERROR: could not make Climo baseline forecast. Make sure at least one forecast"\
            " has been added to the verification first.")
        raise
    climo_fxdates = pd.date_range(climo_sd_ext, climo_ed_ext, freq=vx.inc, name='DateTime', tz=localtz)
    nyrs = relativedelta(climo_ed_ext, climo_sd_ext).years
    if nyrs >= 1:
        climovals = climo_agg.Climo.tolist()*nyrs
    else:
        climovals = climo_agg.Climo.tolist()
    climo_fx = pd.DataFrame(climovals, index=climo_fxdates, columns=['Climo'])
    climo_fx.index = [climo_fx.index[i]+relativedelta(days=1)-relativedelta(months=1) for i in range(len(climo_fx.index))]
    if vx.fxend.tzinfo and vx.fxstart.tzinfo:
        climo_fx = climo_fx[(climo_fx.index <= vx.fxend) & (climo_fx.index >= vx.fxstart)]
    else:
        try:
            climo_fx = climo_fx[(climo_fx.index <= vx.fxend.tz_localize(localtz)) & (climo_fx.index >= vx.fxstart.tz_localize(localtz))]
        except:
            pass
    climo_fx.reset_index(inplace=True)
    climo_fx.rename({'index': 'validtime'}, axis=1, inplace=True)

    #Now set the climo forecast for all horizons
    all_horizons = [h for fcst in vx.forecasts for h in fcst.horizons]
    hs = sorted(list(set(all_horizons)))
    hs = [h for h in hs if h>0]
    if snow_survey:
        hs = [h for h in hs if h <= vx.max_h]
    hzs = hs.copy()
    climo_temp = climo_fx.copy()
    climo_fx["horizon"] = [hs[0]]*len(climo_fx)
    climo_fx['createtime'] = [climo_fx['validtime'][i]-relativedelta(months=climo_fx['horizon'][i]) for i in range(len(climo_fx))]
    hs.pop(0)
    for h in hs:
        climo_temp2 = climo_temp.copy()
        climo_temp2["horizon"] = [h]*len(climo_temp2)
        #TODO: the below line uses "months" in relative delta, but this should depend on inc
        climo_temp2['createtime'] = [climo_temp2['validtime'][i]-relativedelta(months=climo_temp2['horizon'][i]) for i in range(len(climo_temp2))]
        climo_fx = pd.concat([climo_fx, climo_temp2], axis=0)
    climo_fx = climo_fx[['validtime', 'createtime', 'Climo', 'horizon']]
    climo_fx.validtime = pd.to_datetime(climo_fx.validtime)
    climo_fx.createtime = pd.to_datetime(climo_fx.createtime)
    return climo_fx, hzs
    
def get_fx_meta(fxsource, horizontype):
    '''
    For given forecast source (fxsource) and horizontype (short-term or seasonal), returns a
    dictionary of parameters used to build a query to retrieve historical forecast data from the
    Oracle Data Warehouse.
    '''
    import yaml
    auto_path = 'O:\\POOL\\PRIVATE\\Power Management\\ResAdmin\\automated_reports\\verification\\'
    metadata_yaml = "forecast_metadata.yaml"
    with open(auto_path+metadata_yaml) as myf:
        (fx_meta)=yaml.full_load(myf)
        try:
            src_dict = fx_meta[horizontype][fxsource]
        except Exception as e:
            errmsg = "Failed to find metadata from {} for {} {} forecasts."\
                "".format(metadata_yaml, fxsource, horizontype)
            print(errmsg)
            error_email(errmsg=errmsg, e=e, origin_script=__name__)
            raise
    return src_dict

def get_obs_meta(sitename):
    '''
    For given site (sitename), returns a dictionary of parameters used to build a query to
    retrieve streamflow obseration data from the Oracle Data Warehouse.
    '''
    import yaml
    auto_path = 'O:\\POOL\\PRIVATE\\Power Management\\ResAdmin\\automated_reports\\verification\\'
    obsmeta_yaml = "obs_metadata.yaml"
    with open(auto_path+obsmeta_yaml) as obsyf:
        (obs_meta)=yaml.full_load(obsyf)
        try:
            for k in obs_meta:
                if sitename in obs_meta[k]['sites']:
                    obs_src = k
                    break
        except Exception as e:
            errmsg = f"Failed to find obs metadata from {obsmeta_yaml}"
            print(errmsg)
            error_email(e=e, errmsg=errmsg, origin_script=__name__)
            raise
    return obs_src, obs_meta

def get_obs_wrapper(sd, ed, sitename="Ross"):
    '''
    Based on sitename, returns observation data from the appropriate source, and sample temporal resolution code.
    Parameters:
    - sd: Start date. Required, no default.
    - ed: End date. Required, no default.
    - sitename: Site to retrieve obs data for. Default: 'Ross'. 
    '''
    #NOTE: do not resample in this function; data are QC'ed then resampled in add_obs
    #Retrieve database parameters ("metadata") from obs_metadata.yaml via the get_obs_meta method.
    obs_src, meta = get_obs_meta(sitename=sitename)
    samp = 'H' #Assume native resolution is hourly, unless otherwise specified (USACE)
    if obs_src == 'scl':
        sitecode = meta[obs_src]['sites'][sitename]
        obs = get_scl_obs(db='sclba', table='analog', sd=sd+dt.timedelta(hours=1), ed=ed, sitename=sitename, sitecode=sitecode)
    elif obs_src == 'usgs':
        if sitename=='Newhalem' or sitename=='Marblemount':
            get_skagit = False
            obs = get_nhmm_natflow(sd=sd+dt.timedelta(hours=1), ed=ed, get_skagit=get_skagit)
    elif obs_src == 'usace':
        #TODO: allow datastream to be inflow or outflow
        obs = get_usace(station=meta[obs_src]['sites'][sitename], sd=sd, ed=ed, datastream='inflow', per='D')
        obs.rename({'station': sitename}, inplace=True, axis=1)
        samp = 'D'
    else:
        print("ERROR: Unrecognized site / data source!")
        raise

    try:
        obs.index = obs.index.tz_localize('US/Pacific')
    except:
        #Obs are already tz aware
        pass
    obs.index.names = ['DateTime']
    
    if obs_src=='scl':
        return obs[['{}'.format(sitename)]], samp
    return obs, samp

def calc_det_stats(o, f, c=pd.DataFrame()):
    '''
    Calculates deterministic forecast performance stats MAE, RMSE, bias.
    Optionally, calculates skill score relative to c (climatology), if c is not None.
    c should be a dictionary with keys: 'mae', 'rmse', and 'bias'.
    '''
    to_verify = pd.concat([o.copy(),f.copy()], join='inner', axis=1)
    to_verify.dropna(inplace=True)
    cols = to_verify.columns.tolist()
    mae = metrics.mean_absolute_error(to_verify[cols[0]], to_verify[cols[1]])
    rmse = np.sqrt(metrics.mean_squared_error(to_verify[cols[0]], to_verify[cols[1]]))
    bias = to_verify[cols[1]].mean()-to_verify[cols[0]].mean()
    try:
        if len(c):
            return [1-mae/c['MAE_Climatology'].iloc[0], 1-rmse/c['RMSE_Climatology'].iloc[0],
                1-np.abs(bias/c['Bias_Climatology'].iloc[0])]
    except TypeError:
        pass
    except Exception as e:
        print(e)
        raise
    print("mae: {:.3f}, rmse: {:.3f}, bias: {:.3f}".format(mae, rmse, bias))
    return [mae, rmse, bias]


def calc_fx_stats(vx, norm=True, writedata=True, write_dest=None, calc_climo=False, plotstats=True, eval_months=None):
    '''
    Wrapper function to calculate forecast performance errors (or skill scores), which are saved in the
    forecast object attribute stats dictionary. The stats attribute is a dictionary with two keys: 'raw' and 'norm'.
    The values of each of these are dictionaries with month numbers as keys; the '0' key is used for storage of
    forecast performance statistics for all months. The value for each months' key (or the '0' key) is
    a dataframe of forecast performance statistics by horizon.
    Optionally plots the calculated results by horizon, with each forecast source on the same plot.
    Paramters:
    - vx: Verification object. Required, no default.
    - norm: True (default) or False, whether to calculate skill scores or straight error metrics (MAE, RMSE, and bias)
       Determines value of forecast attribute stats_type, equal to 'norm' or 'raw'.
    - writedata: True (default) or False, whether to write the verification results to a csv file when complete.
    - write_dest: Path to write datafile, if writedata = True. Default: None.
    - calc_climo: True or False (default), whether to calculate error statistics for baseline climatology forecast only.
    - plotstats: True (default) or False, whether to generate a plot of forecast performance statistics by horizon
    - eval_months: List of integer months (between 1 and 12) on which to calculate forecast performance statistics, if not all.
       Default: None.
    '''
    #TODO This only works for deterministic. Add probabilistic functionality (CRPS and CRPSS)!
    localtz=vx.localtz
    snow_survey=vx.snow_survey
    bymonth=vx.bymonth
    if writedata and write_dest==None:
        write_dest = f'O:\\POOL\\PRIVATE\\Power Management\\ResAdmin\\automated_reports\\verification\\{vx.site}\\'
        if vx.bymonth:
            bymonth_fn='_bymonth'
        else:
            bymonth_fn=''
        write_dest += f"{vx.fxstart:%Y%m%d%H}-{vx.fxend:%Y%m%d%H}_{vx.site}_{vx.horizontype}{bymonth_fn}\\"
    if not os.path.isdir(write_dest):
        os.makedirs(write_dest)

    if calc_climo:
        if eval_months is not None:
            vx.eval_months = eval_months #Only perform forecast verification on requested months
        else: #No argument passed and therefore no months were requested, so evaluate on all months
            if bymonth:
                #only evaluate months included in all forecasts
                mmins = 0
                mmaxs = np.inf
                for fxo in vx.forecasts:
                    fxd = fxo.get_fx_data()
                    ms = list(set([d.month for d in fxd.validtime]))
                    mmins = max(mmins, min(ms))
                    mmaxs = min(mmaxs, max(ms))
                vx.eval_months = list(np.arange(mmins, mmaxs+1))
            else:
                #Evaluate all months present at once
                vx.eval_months = [0] #This just needs to be a list one item long
        print("Evaluating valid months: ", vx.eval_months)
    else:
        print("Evaluating all months")

    if norm:
        stats_type = 'norm'
        if calc_climo:
            print("ERROR in calc_fx_stats: norm and calc_climo can't both be set to True")
            raise
        else:
            c = vx.climo.stats['raw']
    else:
        stats_type='raw'

    if calc_climo:
        objs = [vx.climo]
        norm = False
        stats_type = 'raw'
        c=pd.DataFrame()
    else:
        objs = vx.forecasts

    for m in vx.eval_months:
        print("Evaluating month {}".format(m))    
        for fxo in objs:
            fxd = fxo.get_fx_data().copy()
            if bymonth:
                fxd = fxd[fxd.validtime.dt.month==m]

            print("Validating {} to {}".format(min(fxd.validtime).strftime("%Y%m%d"), max(fxd.validtime).strftime("%Y%m%d")))
            raw_stats_names = ['MAE_', 'RMSE_', 'Bias_']
            cols = ["horizon"]+[s+fxo.source for s in raw_stats_names]
            hzs = fxo.horizons

            hzs = [h for h in hzs if h>0]
            if snow_survey:
                hzs = [h for h in hzs if h<=vx.max_h]
            #Calculate stats for the forecast object
            fx_stats_holder=[[np.nan, np.nan, np.nan]]*len(hzs)
            for i in range(len(hzs)):
                if hzs[i] > 0 and hzs[i] in fxd.horizon.tolist() and hzs[i] <= 12:
                    fxvx = fxd[fxd.horizon==hzs[i]].copy()
                    if snow_survey:
                        fxvx = fxvx[(fxvx.validtime.dt.month < 10) & (fxvx.validtime.dt.month > fxvx.horizon)]

                    fxvx = fxvx.set_index('validtime')
                    fxvx.index.names= ['DateTime']
                    if not fxvx.index[0].tzinfo:
                        fxvx.index = fxvx.index.tz_localize(localtz)
                    if not vx.obsdata.index[0].tzinfo:
                        vx.obsdata.index = pd.to_datetime(vx.obsdata.index, format="%Y-%m-%d")
                        vx.obsdata.index = vx.obsdata.index.tz_localize(localtz)
                    testdata = pd.concat([vx.obsdata, fxvx], axis=1, join='inner')
                    testdata.to_csv(write_dest+f"{vx.site}_{fxo.source}_HZN{hzs[i]}.csv")
                    if len(testdata)==0:
                        print("Warning: no data found to validate for horizon {}".format(hzs[i]))
                        continue

                    dat_names = testdata.columns.tolist()
                    #TODO: this will definitely not work for probabilistic
                    print("Calculating deterministic stats for source {} for horizon {}".format(fxo.source, hzs[i]))

                    if norm:
                        ch = c[m][c[m].index==hzs[i]].copy()
                    else:
                        ch = None
                    fx_stats_holder[i] = [hzs[i]]+calc_det_stats(testdata[dat_names[0]], testdata[dat_names[2]], ch)
            fs = pd.DataFrame(fx_stats_holder, index=None, columns=cols)
            fs.dropna(inplace=True)
            fs = fs.set_index("horizon")
            fs.sort_index(inplace=True)
            fxo.stats[stats_type][m] = fs

            if bymonth:
                mo_fn_str = "_{}".format(dt.datetime.strftime(dt.datetime.strptime(str(m), "%m"), "%b"))
            else:
                mo_fn_str = ""
            if writedata:
                fn = "{}-{}_{}_{}_{}_fcst_stats.csv".format(vx.fxstart.strftime("%Y%m%d"), vx.fxend.strftime("%Y%m%d"),
                    vx.horizontype, fxo.source.replace(" ", "-"), vx.site, mo_fn_str)
                fs.to_csv(write_dest+fn)
        if plotstats and not calc_climo:
            plot_fx_stats(vx, mons=m, plot_dest=write_dest)

    return

def plot_fx_stats(vx, norm=True, plot_dest=None, plot_name=None, mons=0):
    '''
    Plot forecast performance metrics once they have been calculated.
    Parameters:
    - vx: Verification object. Required, no default.
    - norm: True (default) or False flag indicating whether skill scores or raw forecast metrics
       will be plotted
    - plot_dest: File path where plot should be saved. If none indicated, will be saved to a default
       location. Default: None.
    - plot_name: File name for saved plot. If none indicated, will be saved with a default name.
       Default: None.
    - mons: Calendar months of the year for forecast metrics to be plotted. A zero '0' indicates
       forecast performance statistics should be plotted in bulk (should not be broken out by
       separate calendar months). Default: 0 (all months).
    '''

    #Setup
    if mons:
        mon_abbr = dt.datetime.strftime(dt.datetime.strptime(str(mons), "%m"),"%b")
        bymonth_title = "\nIn the Month of {}".format(mon_abbr)
        bymonth_fn = "_{}".format(mon_abbr)
    else:
        bymonth_title = ""
        bymonth_fn = ""
    if plot_dest == None:
        plot_dest = f'O:\\POOL\\PRIVATE\\Power Management\\ResAdmin\\automated_reports\\verification\\{vx.site}\\'
        #This path may exist already from previous runs
        if not os.path.isdir(plot_dest):
            os.mkdir(plot_dest)
            plot_dest+=f"{vx.fxstart:%Y%m%d%H}-{vx.fxend:%Y%m%d%H}_{vx.site}_{vx.horizontype}{bymonth_fn}\\"
            os.mkdir(plot_dest)
    if norm:
        stats_type = 'norm'
        post, pre = " Skill Score", "Skill"
    else:
        stats_type = 'raw'
        post, pre = "", "Performance"
    titletext = 'Forecast {} for {}\nEvaluated from {} to {}{}'.format(pre, vx.site,
                vx.sd.strftime("%Y-%m-%d"), vx.ed.strftime("%Y-%m-%d"), bymonth_title)
    if plot_name == None:
        plot_name="{}-{}_{}_{}{}_fcst_vx".format(vx.sd.strftime("%Y%m%d"), vx.ed.strftime("%Y%m%d"),
            vx.horizontype, vx.site, bymonth_fn)
    fx_stats_to_join, sources, stat_dfs = [], [], []
    fx_stats = ['MAE', 'RMSE', 'Bias']

    #Compile forecast stats for plot
    for fxo in vx.forecasts:
        fx_stats_to_join += [fxo.stats[stats_type][mons]]
        sources += [fxo.source.capitalize()]
    all_stats = pd.concat(fx_stats_to_join, axis=1, join='outer')
    cols = all_stats.columns.tolist()

    #Make plots of stats by horizon
    fig, axs = plt.subplots(len(fx_stats), 1, sharex=True, figsize=(6,9))
    print(all_stats)
    for s in range(len(fx_stats)):
        plot_cols = [c for c in cols if fx_stats[s] in c]
        plot_dat = all_stats[plot_cols].copy()
        stat_dfs += [plot_dat]
        axs[s].plot(plot_dat, 'o--')
        axs[s].set_ylabel(fx_stats[s]+post)
        axs[s].grid(axis='y', linestyle='-.')
        yticks = ticker.MaxNLocator(6)
        axs[s].yaxis.set_major_locator(yticks)
        axs[s].axhline(linewidth=1, color='k')
    axs[0].legend(labels=sources)
    axs[len(fx_stats)-1].set_xlabel('Horizon ({})'.format(vx.inc))
    fig.suptitle(titletext)
    plt.savefig(plot_dest+plot_name)
    plt.clf()
    return


#TODO calculate P-values for different forecast performance stats

def ross_vx(sd, ed):    
    seas_det = Verification(sitename='Ross', horizontype='seasonal', ed=ed, sd=sd, inc='M', bymonth=False)
    #seas_det.add_forecast(source='snow survey')
    seas_det.add_forecast(source='vendor1')
    seas_det.add_forecast(source='rfc esp')
    seas_det.add_obs()
    seas_det.add_climo()
    calc_fx_stats(seas_det)
    return

def albeni_vx(sd, ed):
    alf_seas_det = Verification(sitename='Albeni', horizontype='seasonal', ed=ed, sd=sd, inc='M', bymonth=False)
    alf_seas_det.add_forecast(source='rfc esp')
    alf_seas_det.add_obs()
    alf_seas_det.add_climo()
    calc_fx_stats(alf_seas_det)
    return

def boundoper_vx(sd,ed):
    oper_seas_det = Verification(sitename='Boundary', horizontype='seasonal', ed=ed, sd=sd, inc='M', bymonth=False)
    oper_seas_det.add_forecast(source='oper')
    oper_seas_det.add_obs()
    oper_seas_det.add_climo()
    calc_fx_stats(oper_seas_det)
    return

if __name__=='__main__':
    sd=dt.datetime(2011,12,1,0,0,0)
    ed=dt.datetime(2021,12,1,0,0,0)
    ross_vx(sd, ed)
