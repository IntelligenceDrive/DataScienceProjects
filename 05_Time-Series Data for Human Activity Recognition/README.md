About data
Smarphones and smartwatches contain tri-axial accelerometers that measure acceleration in all three spatial dimensions. These accelerometers are capable of detecting the orientation of the device, which can provide useful information for activity recognition.
The dataset that we are going to use for this demonstration is sourced from WISDM Lab, Department of Computer & Information Science, Fordham University, Bronx, NY. Note that the file that we are going to use is the raw data file — WISDM_ar_v1.1_raw.txt
This data is collected from 36 different users as they performed some common human activities such as — walking, jogging, ascending stairs, descending stairs, sitting, and standing for specific periods of time. In all cases, the data is collected every 50 millisecond, that is 20 samples per second.
There are total of 5 feature variables— ‘user’, ‘timestamp’, ‘x-axis’, ‘y-axis’, and ‘z-axis’. The target variable is ‘activity’ which we intend to predict.
‘user’ denotes the user ID, ‘timestamp’ is the unix timestamp in nanoseconds, and the rest are the accelerometer readings along the x, y, and z axes/dimensions at a given time.
