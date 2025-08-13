from utils import tactile_reading
pressure, fc, ts = tactile_reading("./ariarecordings/test51/rightPressure.hdf5")
print(fc/(ts[-1]-ts[0]))