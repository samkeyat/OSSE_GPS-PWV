;----------------------------------------------------------------------
; ESMF_curv_to_1deg.ncl
;
; This is an NCL/ESMF template file for regridding from a 
; curvilinear grid to a 1 degree grid. It uses ESMF_regrid 
; to do the regridding.
;
; This script can be easily modified to do use something other than
; a 0.25 degree grid: "1x1", "G64", "5deg", etc.
;
; The curvilinear grid is assumed to be contained in a NetCDF file.
; 
; Search for lines with ";;---Change (likely)" or ";;---Change (maybe)".
; These are the lines you will likely or maybe have to change.
;
; Of course, you'll probably want to change other aspects of this
; code, like the options for plotting (titles, colors, etc).
;
; For more information on ESMF_regrid, see: 
;
;    http://www.ncl.ucar.edu/Document/Functions/ESMF/ESMF_regrid.shtml
;----------------------------------------------------------------------
; This example uses the ESMF application "ESMF_RegridWeightGen" to 
; generate the weights.
;
; For more information about ESMF:
;
;        http://www.earthsystemmodeling.org/
;
; This script uses built-in functions that are only available in 
; NCL V6.1.0-beta and later.
;----------------------------------------------------------------------
;load "$NCARG/lib/ncarg/nclscripts/csm/gsn_code.ncl"
;load "$NCARG/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
;load "$NCARG/lib/ncarg/nclscripts/csm/contributed.ncl"
;load "$NCARG/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin
;---Data file containing source grid
    src_file = "/st6/bayu/ARIZONA_PRO/DART_deterministic/20210703_06Z_mc08_40mem_0035_6cyc_assim/wrfout_d02_2021-07-04_01:00:00"    ;;---Change (likely)
    sfile    = addfile(src_file,"r")

;---Get variable to regrid
    varname = "RAINNC"                      ;;---Change (likely)
    var     = sfile->$varname$(0,:,:)
    src_lat = sfile->XLAT(0,:,:)               ;;---Change (maybe)
    src_lon = sfile->XLONG(0,:,:)               ;;---Change (maybe)

;---Set up regridding options
    Opt                   = True

;---"bilinear" is the default. "patch" and "conserve" are other options.
    Opt@InterpMethod      = "conserve"        ;;---Change (maybe)

    Opt@WgtFileName       = "upscale_weight_WRF2MRMS.AZ+.nc"

    Opt@SrcGridLat        = src_lat           ; source grid
    Opt@SrcGridLon        = src_lon
    Opt@SrcRegional       = True             ;;--Change (maybe)
    Opt@SrcInputFileName  = src_file          ; optional, but good idea
    Opt@SrcMask2D         = where(.not.ismissing(var),1,0) ; Necessary if has
                                                        ; missing values.

    Opt@DstGridType       = "0.01x0.01"            ; Destination grid
    Opt@DstLLCorner       = (/ 31.205,  -115.005 /)   ;;--Change (likely)
    Opt@DstURCorner       = (/ 37.705,  -108.895 /)   ;;--Change (likely)
    Opt@DstRegional       = True                   ;;--Change (maybe)

    Opt@ForceOverwrite    = True
    Opt@PrintTimings      = True
    Opt@Debug             = True

    Opt@SrcForceOverwrite     = True
    Opt@DstForceOverwrite     = True

    rainfall_regrid = ESMF_regrid(var,Opt)     ; Do the regridding

;printVarSummary(rainfall_regrid)
;print(rainfall_regrid&lat)
;sleep(1000)

    ;printVarSummary(var_regrid)
new_file_name = "RAINNC_upscaleWRF2MRMS.AZ+.nc"
system("/bin/rm -f "+new_file_name)  
o = addfile(new_file_name ,"c")
o->RAINNC_regrid = rainfall_regrid;

end
