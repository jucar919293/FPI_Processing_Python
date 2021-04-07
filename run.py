import FPIprocess

# Specify which instrument and which date to process
instr_name = 'minime05'
year = 2013
doy =        273
#year =       2015        
#doy =        166

# Specify where the data are located and where results should be saved
fpi_dir =          ''
results_stub =     'results/'
bw_dir =           ''
x300_dir =         ''

# # Make the call to the processing function
msg = FPIprocess.process_instr(instr_name, year, doy, fpi_dir=fpi_dir,
          bw_dir=bw_dir, x300_dir=x300_dir, results_stub=results_stub,
          send_to_website=False, enable_share=False,
          send_to_madrigal=False, enable_windfield_estimate=False)

# if msg: # if a warning was issued, print it
# 	print msg

# Plot the wind and the temperature using the npz file that was generated above.
# import FPIDisplay

# ((temp_fig,_), (wind_fig,_)) = FPIDisplay.PlotDay('%sminime05_uao_20131001.npz' % results_stub)

# # Save the figures to the results folder
# wind_fig.savefig('%s%s_%i_%03i_winds.png' % (results_stub, instr_name, year, doy))
# temp_fig.savefig('%s%s_%i_%03i_temps.png' % (results_stub, instr_name, year, doy))