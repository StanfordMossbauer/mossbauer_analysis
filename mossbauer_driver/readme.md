# Mossbauer Driver; 

Mossbauer driver is used to read the data from daq system, to some degree , a handmade sql software; 


The primary functions of the driver 
- fetch the L3 processed data from '/data/share' from the daq server; 
- fetch the relevant config files, for example the yml and the filter.npy , gain.npy , dark_2D.npy ; 
- fetch the relevant side streams data if needed 
- fetch the slow control data if needed; 
- basic data analysis functions such as the counts ; 

mossbauer_analysis; 
we have a base library that is also for some other projects; And of course , the camera system will be much bigger than 

We need 
- basic energy resolution
- basic time resolution 
- basic location resolution;
To some degree, a mixed balance of the position/energy/time; 

As to Chiara's DM project, the continuous readout is necessary to us. but unfortunately, this is beyond our ability; 



Current Synchronization is based on two things; 
- The run count and the daq count on the  (The timestamp with counts), but we know that we are running it at 400Hz; 
- The timestamp tagged by the computer system when we received the data; it has a precision of milli seconds based on timestamp, but that is not so precise. 

Suppose that everything is now running under 400Hz; I have to admit that some parameters are hard-coded , which is not a good practice; 


Also ,we have the file name of the , but it is not so reliable; and aside from L1, every file; 

Another issue that is worth mentioning is that the L1 could not be efficiently loaded, and it is quite large, so it is possible that when we fetch the L1, we have to go to the DAQ computer; 

the daq computer is 

elm only handles the 

I do not want to fit this driver into the driver that already exist, because my driver will be much larger than anything else. 

The data analysis 


The data acquistion 


The first task to be finished is to 

get the file name for the files we want , and then map it to the 
 