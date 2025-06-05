;;; This script is to extract variables that are not given by wrfout from DETERMINISTIC RUN
;;; Usage: For deterministic run assim and noassim
;;; Created by C. Bayu Risanto, S.J. (16 February 2024)
;;********************************************************************************************

time = 0

dart_dir = "20210901_06Z_mc08_40mem_0035_6cyc_noassim"
dir1 = "/home/bayu/DATA/ARIZONA_PRO/DART_deterministic/" + dart_dir + "/" 
files = systemfunc("ls " + dir1 + "wrfout_d02_*")
fout_pre = systemfunc("ls " + dir1 + "wrfout_d02_* | xargs -n 1 basename | cut -f2-4 -d_")
nf = dimsizes(files)
dirout = "/home/bayu/DATA/ARIZONA_PRO/WRF_EXTR/WRF_OUT/" + dart_dir + "/" 

opt = True
ii = 0

do while(ii.le.nf-1)

    wcday = systemfunc("date")
    a = addfile(files(ii),"r") 

    print(""+files(ii))

    fout1 = dirout + "/ncl-2D_" + fout_pre(ii) + ".nc"
    system("/bin/rm -f " + fout1)   ; remove any pre-existing file
  
    ctt1 = wrf_user_getvar(a,"ctt",0)      ; cloud top temperature
 
    mccll = wrf_user_getvar(a,"cape_2d",0) ; mcape/mcin/lcl/lfc
    mcape1 = mccll(0,:,:)
    mcin1  = mccll(1,:,:)
    lcl1   = mccll(2,:,:)
    lfc1   = mccll(3,:,:)
    delete(mccll)
 
    pw1 = wrf_user_getvar(a,"pw",0)    ; precipitable water

    td1 = wrf_user_getvar(a,"td2",0)   ; 2m dew point temperature
    
    uvmet = wrf_user_getvar(a,"uvmet10",0) 
    umet = uvmet(0,:,:)
    vmet = uvmet(1,:,:)
    delete(uvmet)
    wspd = wind_speed(umet, vmet)
    
    p = wrf_user_getvar(a,"pressure",0)  ; total pressure
    u = wrf_user_getvar(a,"ua",0)        ; u averaged to mass points
    v = wrf_user_getvar(a,"va",0)        ; v averaged to mass points
    u1_850 = wrf_user_interp_level(u, p, 850, opt)
    v1_850 = wrf_user_interp_level(v, p, 850, opt)
    u1_500 = wrf_user_interp_level(u, p, 500, opt)
    v1_500 = wrf_user_interp_level(v, p, 500, opt)
    delete(p)
    delete(u)
    delete(v)

    print(""+fout1)

    nc1 = addfile(fout1 ,"c")  ; open output netCDF file for ctt, cape, pwv 
    nc1->ctt   = ctt1
    nc1->pw    = pw1
    nc1->td2   = td1
    nc1->mcape = mcape1
    nc1->mcin  = mcin1
    nc1->lcl   = lcl1
    nc1->lfc   = lfc1 
    nc1->umet  = umet
    nc1->vmet  = vmet
    nc1->wspd  = wspd
    nc1->u_850 = u1_850
    nc1->v_850 = v1_850
    nc1->u_500 = u1_500
    nc1->v_500 = v1_500

    delete(ctt1)
    delete(pw1)
    delete(td1)
    delete(mcape1)
    delete(mcin1)
    delete(lcl1)
    delete(lfc1)
    delete(umet)
    delete(vmet)
    delete(wspd)
    delete(u1_850)
    delete(v1_850)
    delete(u1_500)
    delete(v1_500)
    
    ;wallClockElapseTime(wcday, "one day", 0)

    delete(nc1)
    print(" ")     
                            
    ii = ii + 1

end do

exit

