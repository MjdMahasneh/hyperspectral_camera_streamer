import socket
import struct
import numpy as np
from threading import Thread
from collections import deque
import time
from pathlib import Path
import json


# DEFINES
EXAMPLE_USE_MANUAL_ROI = False
EXAMPLE_USE_FRAME_TRIGGER_MODE = False
EXAMPLE_USE_ACQUISITION_BURST_MODE = False
EXAMPLE_USE_CALIBRATED_ROI = False
EXAMPLE_USE_PIXEL_PACKING = False
EXAMPLE_AUTOSELECT_MANUAL = False
EXAMPLE_ENABLE_SAVE_USER_CONFIG = False
EXAMPLE_ENABLE_SPATIAL_BINNING = False
EXAMPLE_DIFFERENTIAL_TRIGGER = False
EXAMPLE_BLOCK_TCP_UNTIL_WARMUP = False

# For this to work, a classifier must be trained with blackstudio and must be transferred to the camera
# also the camera mode must be set the same as used for training the classifier
EXAMPLE_ON_CAMERA_CLASSIFICATION_ENABLED = False

### COMMANDS FOR SENSOR COMMUNCATION ###

#MODE
m_SET = 0
m_GET = 1

#FUNCTIONS
f_GAIN = 1
f_EXPOSURE = 2
f_TEMPERATURE = 5
f_RECV_CONFIG = 6
f_STARTSTREAMING = 7
f_FRAMERATE = 9
f_MODE = 10
f_SERIALNR = 11
f_FEATURE_SUPPORTED = 13
f_CAM_GET_ROI_LIMITS = 14
f_CAM_SET_ROI = 15
f_CAM_GET_ROI_LIMITS = 14
f_CAM_SET_ROI = 15
f_SET_CALIBRATED_ROI_START = 16
f_SET_CALIBRATED_ROI_END = 17
f_GET_CURRENT_RESOLUTION = 18
f_GET_CURRENT_MAXFPS = 19
f_PIXEL_FORMAT = 20
f_AUTO_SELECT_MANUAL_ROI = 21
f_PREPROCESS = 23
f_TCP_BLOCK_SENDOUT = 24
f_CAM_GET_STATUS = 25

f_INPUT_TRIGGER_DIVIDER = 41
f_INPUT_TRIGGER_MODE = 42
f_CAM_AUTO_MAX_EXPOSURE   = 43
f_INPUT_TRIGGER_FREQUENCY = 44
f_OUTPUT_TRIGGER_MODE = 45
f_ACQUISITION_BURST_LENGTH = 46

f_CAM_SAVE_USER_CONFIG = 52

f_CAM_OUTPUT_TRIGGER_PIN_MODE = 56
f_CAM_INPUT_TRIGGER_PIN_MODE = 57

#PARAMETER NONE
p_NONE = 0


# Pixel formats
class pixel_formats:
    MONO10 = 0
    MONO8 = 1
    MONO10_4_5 = 2
    MONO10_2_3 = 3
    CLASSES_COLOUR = 4
    
class spatial_binning_modes:
    NO_BINNING = 1
    BINNING_2 = 2
    

class CamConfigMode:
    def __init__(self, data):
        self.NAME = data[0].decode("utf-8").rstrip('\0')
        self.CAM_MODE, self.MAX_VALUE, self.WHITE_POINT, self.SPATIAL_PIXEL, self.SPECTRAL_BANDS, \
            self.SPECTRAL_MIN, self.SPECTRAL_MAX, self.BLUE_CHANNEL_BAND_STD, self.GREEN_CHANNEL_BAND_STD,\
            self.RED_CHANNEL_BAND_STD,self.MAX_FPS = data[1:]
class CamConfigStruct:
    def __init__(self, data):
        header = struct.unpack("100si", data[:struct.calcsize("100si")])
        self.name = header[0].decode("utf-8").rstrip('\0')
        self.available_modes = header[1]
        self.modes = []
        self.manual_roi_mode = None
        i = 0
        for mode in struct.iter_unpack("100s11i", data[struct.calcsize("100si"):]):
            self.modes.append(CamConfigMode(mode))

            # 32 SPECTRAL_BANDS is our manual ROI mode
            # in reality SPECTRAL_BANDS will be set according to ROI settings
            if(self.modes[-1].SPECTRAL_BANDS == 32):
                self.manual_roi_mode = i

            i = i + 1
            if (i == self.available_modes):
                break

class HAIP_BlackIndustry:

    # TRIGGER INPUT MODES

    # Default mode; Camera handles framerate
    MASTER_MODE = 0
    # Capturing of frames is triggered by an outbound trigger signal 
    FRAME_TRIGGER_MODE = 1
    # If an outbound trigger signal is received, a number of frames is recorded 
    ACQUISITION_BURST_MODE = 2

    # TRIGGER OUTPUT MODES
    TRIGGER_FROM_SENSOR = 0
    RISING_EDGE_TRIGGER_FROM_EXTERNAL = 1
    FALLING_EDGE_TRIGGER_FROM_EXTERNAL = 2
    
    # TRIGGER PIN MODES
    SINGLE_ENDED_5V = 0
    SINGLE_ENDED_24V = 1
    DIFFERENTIAL = 2
    
    def __init__(self, TCP_PORT = 7892, BUFFER_SIZE=2048):

        # TCP configuration
        self.__TCP_IP = ""
        self.__TCP_PORT = TCP_PORT
        self.__connection = None
        self.__BUFFER_SIZE = BUFFER_SIZE

        self.__mode = 0

        self.__livestreamActive = False

        self.__worker = None
        self.__q = None

        self.manual_roi_spectral_bands = 32

        self.FRAME_HEADER_SUPPORT = 0
        self.MANUAL_ROI_SUPPORT = 0
        self.TEMP_FRAME_SUPPORT = 0
        self.TRIGGER_SUPPORT = 0
        self.CALIBRATED_ROI_SUPPORT = 0
        self.SELECT_PIXEL_PACKING = 0

        self.DIFFERENTIAL_TRIGGER_SUPPORT = 0

        self.pixel_format = 0

    def __del__(self):
        if(not self.__connection == None):
            self.__stopStreaming()
            self.__connection.close()


    ######### HELPER FUNCTIONS ###############

    def __startConnection(self, timeout=1):
        if(self.__TCP_IP == ""):
            print("Please init camera before first call.")
            return False
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.settimeout(timeout)
        try:
            connection.connect((self.__TCP_IP, self.__TCP_PORT))
        except Exception as e:
            connection.close()
            print("Connection failed!")
            return False
        return connection

    def __validateConnection(self):
        connection = self.__startConnection()
        if (connection):
            connection.close()
            return True
        else:
            print("Connection failed.")
            return False

    def __headerInformation(self):
        header = None
        while(header is None):
            try:
                header = self.__connection.recv(32)
            except socket.timeout:
                print("Timeout header. Keep trying.")
        header = struct.unpack('HHHHHHHHHHHHHHHH', header)
        spatial_resolution = header[0]
        spectral_resolution = header[1]
        return spatial_resolution, spectral_resolution

    def __receiveImage(self, spectral, spatial):
        MSGLEN = spatial*spectral*2
        datatype = "uint16"
        if (self.pixel_format == pixel_formats.MONO8):
            MSGLEN = int(spatial * spectral * 8.0/8.0)
            datatype = "uint8"
        if (self.pixel_format == pixel_formats.MONO10_4_5):
            MSGLEN = int(spatial * spectral * 10.0 / 8.0)
        if (self.pixel_format == pixel_formats.MONO10_2_3):
            MSGLEN = int(spatial * spectral * 12.0 / 8.0)
        if (self.pixel_format == pixel_formats.CLASSES_COLOUR):
            MSGLEN = int(spatial * 3.0)
            datatype = "uint8"

        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            try:
                chunk = self.__connection.recv(min(MSGLEN - bytes_recd, self.__BUFFER_SIZE))
            except socket.timeout:
                print("Timeout while receiving data. Checking if camera is still connected.")
                return None
            bytes_recd = bytes_recd + len(chunk)
            chunks.append(chunk)

        if (self.pixel_format == pixel_formats.MONO10_4_5):
            packed_values = np.frombuffer(b''.join(chunks), dtype=np.dtype("uint8"), offset=0)
            packed_values = packed_values.reshape(-1, 5).astype("uint16")

            unpacked = np.empty(((spatial * spectral) // 4, 4), dtype="uint16")

            # lsb packed
            unpacked[:, 0] = packed_values[:, 0] | ((packed_values[:, 1] & 0x03) << 8)
            unpacked[:, 1] = ((packed_values[:, 1] >> 2) & 0x3F) | ((packed_values[:, 2] & 0x0F) << 6)
            unpacked[:, 2] = ((packed_values[:, 2] >> 4) & 0x0F) | ((packed_values[:, 3] & 0x3F) << 4)
            unpacked[:, 3] = (packed_values[:, 4] & 0xFF) << 2 | (packed_values[:, 3] >> 6)

            arr = unpacked.reshape(spatial, spectral)

        elif (self.pixel_format == pixel_formats.MONO10_2_3):
            packed_values = np.frombuffer(b''.join(chunks), dtype=np.dtype("uint8"), offset=0)
            packed_values = packed_values.reshape(-1, 3).astype("uint16")

            unpacked = np.empty(((spatial * spectral) // 2, 2), dtype="uint16")

            # lsb packed
            unpacked[:, 0] = packed_values[:, 0] << 2 | (packed_values[:, 1] & 0x03)
            unpacked[:, 1] = packed_values[:, 2] << 2 | (packed_values[:, 1] >> 4)

            arr = unpacked.reshape(spatial, spectral)

        elif (self.pixel_format == pixel_formats.CLASSES_COLOUR):
            arr = np.frombuffer(b''.join(chunks), dtype=datatype).reshape(spatial, 3)
        
        else:
            arr = np.frombuffer(b''.join(chunks), dtype=datatype).reshape(spatial, spectral)
        return arr

    def __recvImageWorker(self, q):
        spectral, spatial = self.getCurrentResolution()
            
        # for backwards compatibility
        if(spectral == 0 or spatial == 0):
            spectral, spatial = self.cam_config_struct.modes[self.__mode].SPECTRAL_BANDS, self.cam_config_struct.modes[self.__mode].SPATIAL_PIXEL
        self.__startStreaming()
        self.__headerInformation()
        while self.__livestreamActive:
            im = self.__receiveImage(spectral, spatial)
            if (im is None):
                if(self.__livestreamActive):
                   temp, stream = self.getStatus()
                   if(stream is None):
                        print("Camera is not connected. Closing connection.")
                        self.__livestreamActive = False
                        break
                   elif (stream == 0):
                       print("Camera is connected, but not streaming. Closing connection.")
                       self.__livestreamActive = False
                       break
                   else:
                      print("Camera is connected and streaming. Waiting for more data.")
                else:
                    print("Timeout and livestream should be stopped. Closing connection.")
            else:
                q.append(im)
        self.__stopStreaming()

    def __setCommand(self, f, p1, p2, timeout=0.2):
        connection = self.__startConnection(timeout)
        if (connection):
            message = struct.pack('<bbhII', 0, m_SET, f, p1, p2)
            connection.send(message)
            connection.recv(struct.calcsize('<bbhII'))
            connection.close()
        else:
            print("Could not connect!")

    def __getCommand(self, f, p1, p2):
        connection = self.__startConnection()
        if(connection):
            message = struct.pack('<bbhII', 0, m_GET, f, p1, p2)
            connection.send(message)
            data = connection.recv(self.__BUFFER_SIZE)
            unpacked = struct.unpack('<bbhii', data)
            connection.close()
            return unpacked
        else:
            print("Could not connect!")


    def __startStreaming(self):
        self.__connection = self.__startConnection(timeout = 4)
        if(self.__connection):
            message = struct.pack('<bbhII', 0, m_SET, f_STARTSTREAMING, 0, p_NONE)
            self.__connection.send(message)
        else:
            print("Could not connect!")

    def __stopStreaming(self):
        self.__connection.close()
        self.__connection = None

    ############ init #################

    def init(self, TCP_IP):
        self.__TCP_IP = TCP_IP
        connection = self.__startConnection()
        if(connection):
            message = struct.pack('<bbhII', 0, m_GET, f_RECV_CONFIG, 0, 0)
            connection.send(message)

            cam_config_struct_data = connection.recv(struct.calcsize("100si") + 12*struct.calcsize("100s11i"))
            self.cam_config_struct = CamConfigStruct(cam_config_struct_data)

            if(self.cam_config_struct.manual_roi_mode != None):
                self.cam_config_struct.modes[self.cam_config_struct.manual_roi_mode].SPECTRAL_BANDS = self.manual_roi_spectral_bands

            data = connection.recv(self.__BUFFER_SIZE)
            unpacked = struct.unpack('<bbhII', data)
            connection.close()

            feature_support = self.getFeatureSupport()
            self.FRAME_HEADER_SUPPORT = feature_support & 1
            self.MANUAL_ROI_SUPPORT = feature_support & (1 << 1)
            self.TEMP_FRAME_SUPPORT = feature_support & (1 << 2)
            self.TRIGGER_SUPPORT = feature_support & (1 << 3)
            self.CALIBRATED_ROI_SUPPORT = feature_support & (1 << 4)
            self.SELECT_PIXEL_PACKING = feature_support & (1 << 5)
            self.SAVE_USER_CONFIG_SUPPORT = feature_support & (1 << 7)
            self.ON_CAMERA_CLASSIFIER_SUPPORT = feature_support & (1 << 8)
            self.SPATIAL_BINNING_SUPPORT = feature_support & (1 << 9)
            self.DIFFERENTIAL_TRIGGER_SUPPORT = feature_support & (1 << 11)

            self.getpixel_format()
            self.getMode()
        print("Could not connect!")

    ############ Get functions ############

    def getStatus(self):
        unpacked = self.__getCommand(f_CAM_GET_STATUS, p_NONE, p_NONE)
        if(unpacked is None):
            print("Could not connect!")
            return None,None
        else:
            return (unpacked[3] & 0b1) , (unpacked[3] & 0b10) >> 1


    def getCurrentResolution(self, mode=None):
        if(mode is None):
            mode = self.__mode
        unpacked = self.__getCommand(f_GET_CURRENT_RESOLUTION, self.cam_config_struct.modes[mode].CAM_MODE, p_NONE)
        return unpacked[3], unpacked[4]
    def getCurrentMaxFPS(self, mode=None):
        if (mode is None):
            mode = self.__mode
        unpacked = self.__getCommand(f_GET_CURRENT_MAXFPS, self.cam_config_struct.modes[mode].CAM_MODE, p_NONE)
        return unpacked[3]
    def getCurrentMaxExposure(self, mode=None):
        if (mode is None):
            mode = self.__mode
        unpacked = self.__getCommand(f_GET_CURRENT_MAXFPS, self.cam_config_struct.modes[mode].CAM_MODE, p_NONE)
        return unpacked[4]

    def getFeatureSupport(self):
        unpacked = self.__getCommand(f_FEATURE_SUPPORTED, p_NONE, p_NONE)
        return unpacked[3]
    def getModeConfig(self):
        return self.cam_config_struct

    def getGain(self):
        unpacked = self.__getCommand(f_GAIN, p_NONE, p_NONE)
        return unpacked[3]

    def getFPS(self):
        unpacked = self.__getCommand(f_FRAMERATE , p_NONE, p_NONE)
        return unpacked[3]

    def getExposure(self):
        unpacked = self.__getCommand(f_EXPOSURE, p_NONE, 1)
        return unpacked[3]

    def getpixel_format(self):
        unpacked = self.__getCommand(f_PIXEL_FORMAT, p_NONE, p_NONE)
        self.pixel_format = unpacked[3]
        return unpacked[3]

    def getMode(self):
        unpacked = self.__getCommand(f_MODE, p_NONE, p_NONE)
        sensor_mode = unpacked[3]
        for i in range(self.cam_config_struct.available_modes):
                if (sensor_mode == self.cam_config_struct.modes[i].CAM_MODE):
                    self.__mode = i
                    break
        return self.__mode
    def getTemperature(self):
        unpacked = self.__getCommand(f_TEMPERATURE, p_NONE, p_NONE)
        divider = unpacked[4]
        if (divider == 0):
            divider = 10
        return unpacked[3] / float(divider)

    def getSerialNr(self):
        connection = self.__startConnection()
        if connection:
            message = struct.pack('<bbhII', 0, m_GET, f_SERIALNR, p_NONE, p_NONE)
            connection.send(message)
            serialnr = connection.recv(20)
            connection.close()
            return serialnr.decode().rstrip('\x00')
        else:
            return "N/A"

    def getVersionNr(self):
        connection = self.__startConnection()
        if connection:
            message = struct.pack('<bbhII', 0, m_GET, f_SERIALNR, 1, p_NONE)
            connection.send(message)
            versionnr = connection.recv(20)
            connection.close()
            return versionnr.decode().rstrip('\x00')
        else:
            return "N/A"

    def getROILimits(self):
        connection = self.__startConnection()
        if connection:
            message = struct.pack('<bbhII', 0, m_GET, f_CAM_GET_ROI_LIMITS, p_NONE, p_NONE)
            connection.send(message)
            roiLimits_data = connection.recv(
                struct.calcsize("ii??") * 129)  # 129 should be enough for swir and swirmax
            connection.close()

            roi_limits = list(struct.iter_unpack("ii??", roiLimits_data))
            self.manual_roi_spectral_bands = len([region for region in roi_limits if region[2]]) * 4
            self.cam_config_struct.modes[self.cam_config_struct.manual_roi_mode].SPECTRAL_BANDS = self.manual_roi_spectral_bands
            return roi_limits
        else:
            return None

    def getCalibratedRoi(self):
        list_regions = []
        for i in range(8):
            _,_,_,_,start = self.__getCommand(f_SET_CALIBRATED_ROI_START, i , 0)
            _,_,_,_,end = self.__getCommand(f_SET_CALIBRATED_ROI_END, i, 0)
            if(start != 0 and end != 0):
                list_regions.append((start,end))
        # returns a list of tuple (start, end)
        return list_regions

    def getSaveUserConfig(self):
        unpacked = self.__getCommand(f_CAM_SAVE_USER_CONFIG, p_NONE, p_NONE)
        return unpacked[3]

    def getSpatialBinning(self):
        unpacked = self.__getCommand(f_PREPROCESS, 2, p_NONE)
        return unpacked[4]

    ############ Trigger Specific Get Functions #######

    def getInputTriggerMode(self):
        unpacked = self.__getCommand(f_INPUT_TRIGGER_MODE, p_NONE, p_NONE)
        return unpacked[3]

    def getInputTriggerFrequency(self):
        unpacked = self.__getCommand(f_INPUT_TRIGGER_FREQUENCY, p_NONE, p_NONE)
        return unpacked[3]
    
    def getInputTriggerDividerValue(self):
        unpacked = self.__getCommand(f_INPUT_TRIGGER_DIVIDER, p_NONE, p_NONE)
        return unpacked[3]
        
    def getAcquisitionBurstLength(self):
        unpacked = self.__getCommand(f_ACQUISITION_BURST_LENGTH, p_NONE, p_NONE)
        return unpacked[3]
        
    def getOutputTriggerMode(self):
        unpacked = self.__getCommand(f_OUTPUT_TRIGGER_MODE, p_NONE, p_NONE)
        return unpacked[3]
    def getOutputTriggerMode(self):
        unpacked = self.getCommand(f_OUTPUT_TRIGGER_MODE, p_NONE, p_NONE)
        self.signal_OutputTriggerMode.emit(unpacked[3])
        return unpacked[3]


    def getOutputTriggerPinMode(self):
        unpacked = self.getCommand(f_CAM_OUTPUT_TRIGGER_PIN_MODE, p_NONE, p_NONE)
        self.signal_OutputTriggerPinMode.emit(unpacked[3])
        return unpacked[3]

    def getInputTriggerPinMode(self):
        unpacked = self.getCommand(f_CAM_INPUT_TRIGGER_PIN_MODE, p_NONE, p_NONE)
        self.signal_InputTriggerPinMode.emit(unpacked[3])
        return unpacked[3]
    ############ Set functions ############

    def setSpatialBinning(self, spatial_binning_mode):
        self.__setCommand(f_PREPROCESS, 2, spatial_binning_mode, timeout=2)

    def auto_select_manual_roi(self, number):
        self.__setCommand(f_AUTO_SELECT_MANUAL_ROI, number, 0)
        self.getROILimits()

    def writeROIsToCamera(self, list_regions):
        # sending 0 clears all ROIs on camera
        self.__setCommand(f_CAM_SET_ROI, 0, p_NONE)

        # now activating each roi (offset by 1)
        for region in list_regions[:-1]:
            self.__setCommand(f_CAM_SET_ROI, region+1, p_NONE)

        # sending last roi combined with sending 1 for value2 makes it take effect on server side
        self.__setCommand(f_CAM_SET_ROI, list_regions[-1] + 1, 1, timeout=2)
        self.manual_roi_spectral_bands = len(list_regions)*4
        self.cam_config_struct.modes[self.cam_config_struct.manual_roi_mode].SPECTRAL_BANDS = self.manual_roi_spectral_bands

    def setCalibratedRoi(self, list_regions):
        counter = 0
        # list_regions should be a list of tuple of start and end
        for start, end in list_regions:
            self.__setCommand(f_SET_CALIBRATED_ROI_START, counter, int(start))
            self.__setCommand(f_SET_CALIBRATED_ROI_END, counter, int(end))
            counter = counter + 1
        for i in range(counter,8):
            self.__setCommand(f_SET_CALIBRATED_ROI_START, counter, 0)
            self.__setCommand(f_SET_CALIBRATED_ROI_END, counter, 0)
            counter = counter + 1

        # sending 100 for value1 makes it take effect on server side
        self.__setCommand(f_SET_CALIBRATED_ROI_START, 100, 0, timeout=2)

    def setTCPBlockSendout(self, value):
        self.__setCommand(f_TCP_BLOCK_SENDOUT, value, p_NONE)

    def setMode(self, mode):
        self.__setCommand(f_MODE, self.cam_config_struct.modes[mode].CAM_MODE, p_NONE, timeout=2)
        self.__mode = mode

    def setGain(self, gain):
        self.__setCommand(f_GAIN, gain, p_NONE)

    def setExposure(self, exposure):
        self.__setCommand(f_EXPOSURE, exposure, 1)

    def setFPS(self, value):
        self.__setCommand(f_FRAMERATE, value, p_NONE)

    def set_pixel_format(self, pixel_format):
        self.pixel_format = pixel_format
        self.__setCommand(f_PIXEL_FORMAT, pixel_format, p_NONE, timeout=2)

    def setSaveUserConfig(self, value):
        self.__setCommand(f_CAM_SAVE_USER_CONFIG, value, p_NONE)
        
    ############ Trigger Specific Set Functions #######

    def setInputTriggerMode(self, value):
        self.__setCommand(f_INPUT_TRIGGER_MODE, value, p_NONE, timeout=2)
    
    def setInputTriggerDividerValue(self, value):
        self.__setCommand(f_INPUT_TRIGGER_DIVIDER, value, p_NONE)
        
    def setAcquisitionBurstLength(self, value):
        self.__setCommand(f_ACQUISITION_BURST_LENGTH, value, p_NONE)
        
    def setOutputTriggerMode(self, value):
        self.__setCommand(f_OUTPUT_TRIGGER_MODE, value, p_NONE)
    
    def setOutputTriggerPinMode(self, value):
        self.setCommand(f_CAM_OUTPUT_TRIGGER_PIN_MODE, value, p_NONE)
    
    def setInputTriggerPinMode(self, value):
        self.setCommand(f_CAM_INPUT_TRIGGER_PIN_MODE, value, p_NONE)

    
    ############ Livestream functions ############

    def startCameraStream(self):
        self.__livestreamActive = True
        self.__q = deque(maxlen=2)
        self.__worker = Thread(target=self.__recvImageWorker, args=(self.__q,), daemon=True)
        self.__worker.start()

    def stopCameraStream(self):
        self.__livestreamActive = False
        self.__worker.join()

    def getImage(self):
        if(self.__q):
            return self.__q.pop()
        else:
            return None



def example():
    #creating camera object
    camera = HAIP_BlackIndustry()

    # init connection camera using standard IP 192.168.7.1
    camera.init("192.168.7.1")

    # we can get the mode config struct and see available modes
    config = camera.getModeConfig()
    print(config.name)
    print("Modes:")
    for i in range(config.available_modes):
        print(config.modes[i].NAME)

    print("Current mode:", camera.getMode())

    # you can use these model specific variables so your tools work with different modes and models of us
    print("Printing Mode 0 in detail:")
    print("MAX_VALUE: ", config.modes[0].MAX_VALUE)
    print("SPATIAL_PIXEL: ", config.modes[0].SPATIAL_PIXEL)
    print("SPECTRAL_BANDS: ", config.modes[0].SPECTRAL_BANDS)
    print("SPECTRAL_MIN: ", config.modes[0].SPECTRAL_MIN)
    print("SPECTRAL_MAX: ", config.modes[0].SPECTRAL_MAX)
    print("MAX_FPS: ", config.modes[0].MAX_FPS)

    camera.set_pixel_format(pixel_formats.MONO10)
    camera.setMode(0) # use first mode

    camera.setFPS(100) # 100fps
    camera.setExposure(9000) # 9ms exposure time
    camera.setGain(2) # 2% sensor analog gain

    # receive and print serialnr
    print("Serialnr:", camera.getSerialNr())

    if(EXAMPLE_USE_MANUAL_ROI and camera.MANUAL_ROI_SUPPORT):
        # please note that before receiving ROI limits, SPECTRAL_BANDS for ROI mode will always be 32
        # even when in reality it is not
        print("ROI Mode SPECTRAL_BANDS before receiving ROI limits: ", config.modes[2].SPECTRAL_BANDS)
        # receive roi limits and current status
        for index, region in enumerate(camera.getROILimits()):
            # each entry is a tuple of 4
            # also we will need the index for later
            min, max, used, active = region
            if(used):
                if(active):
                    print("Region", index, str(min)+"nm", "-",str(max)+"nm", "- ACTIVE")
                else:
                    print("Region", index, str(min)+"nm", "-", str(max)+"nm")

        # add index of rois
        # You have to select a minimum of 8 regions and a maximum of 8 combined region areas.
        # for example: region index 30,31,32,33,34 and region index 54,55,56,57,58
        # this would be 10 regions and 2 combined region areas
        camera.writeROIsToCamera([30,31,32,33,34,54,55,56,57,58])

        # !!! DEPRECATED - better to use functions below setMode to receive correct resolution and possible fps !!!
        # Now that we have received or set ROI limits the SPECTRAL_BANDS value is updated
        # you get 4 pixel for every region you select.
        print("ROI Mode SPECTRAL_BANDS after receiving ROI limits: ", config.modes[2].SPECTRAL_BANDS)

        # now we switch to manual roi mode for receive
        camera.setMode(2)

        # the SPECTRAL_BAND value from mode config cannot be used and will always be 32 for manual ROI modes
        # these functions will give you the correct resolution and fps possible with these roi settings
        # print("Current Mode Maximum FPS:", camera.getCurrentMaxFPS())
        # print("Current Mode Resolution:", camera.getCurrentResolution())

    if (EXAMPLE_USE_CALIBRATED_ROI and camera.CALIBRATED_ROI_SUPPORT):

        current_calibrated_rois = camera.getCalibratedRoi()
        for start, end in current_calibrated_rois:
            print("ROI", start, "nm - ", end, "nm")

        wanted_rois = [(1050, 1150), (1350, 1400)]
        camera.setCalibratedRoi(wanted_rois)

        # now we switch to calibrated roi mode for receive
        camera.setMode(3)

        # the SPECTRAL_BAND value from mode config cannot be used and will always be 40 for Calibrated ROI modes
        # these functions will give you the correct resolution and fps possible with these roi settings
        #print("Current Mode Maximum FPS:", camera.getCurrentMaxFPS())
        #print("Current Mode Resolution:", camera.getCurrentResolution())

    if (EXAMPLE_AUTOSELECT_MANUAL and camera.MANUAL_ROI_SUPPORT and camera.CALIBRATED_ROI_SUPPORT):
        # this will try to select the 10 best manual ROIs based on your calibrated roi settings
        camera.auto_select_manual_roi(10)

        # now print the result
        for index, region in enumerate(camera.getROILimits()):
            min, max, used, active = region
            if(used):
                if(active):
                    print("Region", index, str(min)+"nm", "-",str(max)+"nm", "- ACTIVE")

        # now we switch to manual roi mode for receive
        camera.setMode(2)

    if (EXAMPLE_USE_FRAME_TRIGGER_MODE and camera.TRIGGER_SUPPORT):
        # USE FRAME TRIGGER MODE

        camera.setInputTriggerMode(HAIP_BlackIndustry.FRAME_TRIGGER_MODE)
        camera.setInputTriggerDividerValue(1)
                
        # after trigger mode changes, fps and exposure parameters are reset
        # so set these parameters again
        camera.setFPS(100) # 100fps
        camera.setExposure(1000) # 1ms exposure time
        
        print("INPUT FREQUENCY AT TRIGGER INPUT IS: ", camera.getInputTriggerFrequency())
        print("RESULTING FRAMERATE IS: ", camera.getFPS())
        print("INPUT TRIGGER MODE IS: ", camera.getInputTriggerMode())
        print("INPUT TRIGGER DIVIDER VALUE IS: ", camera.getInputTriggerDividerValue())

        
        if(camera.getInputTriggerFrequency()<10):
            camera.setInputTriggerMode(HAIP_BlackIndustry.MASTER_MODE)    
            print("INPUT FREQUENCY AT TRIGGER INPUT IS TOO LOW. RESETTING TO MASTER MODE")

    
    if (EXAMPLE_USE_ACQUISITION_BURST_MODE and camera.TRIGGER_SUPPORT):
        # USE ACQUISITION BURST MODE
        
        camera.setInputTriggerMode(HAIP_BlackIndustry.ACQUISITION_BURST_MODE)
        camera.setAcquisitionBurstLength(2)
        
        # after trigger mode changes, fps and exposure parameters are reset
        # so set these parameters again
        camera.setFPS(100) # 100fps
        camera.setExposure(1000) # 1ms exposure time
        
        print("ACQUISITION BURST LENGTH IS: ", camera.getAcquisitionBurstLength())
        print("TRIGGER MODE IS: ", camera.getInputTriggerMode())

        
        if(camera.getInputTriggerFrequency()>100):
            camera.setInputTriggerMode(HAIP_BlackIndustry.MASTER_MODE)    
            print("INPUT_FREQUENCY AT TRIGGER INPUT IS TOO HIGH FOR BURST MODE. RESETTING TO MASTER MODE")

    if (EXAMPLE_USE_PIXEL_PACKING and camera.SELECT_PIXEL_PACKING):
        # set pixel packing format to save ethernet bandwidth
        camera.set_pixel_format(pixel_formats.MONO10_4_5)

    if (EXAMPLE_ENABLE_SAVE_USER_CONFIG and camera.SAVE_USER_CONFIG_SUPPORT):
        # Enables that camera configuration (Exposure, FPS, Gain, mode etc.) are saved on the camera and loaded at restart
        camera.setSaveUserConfig(True)
    
    if (EXAMPLE_ENABLE_SPATIAL_BINNING and camera.SPATIAL_BINNING_SUPPORT):
        current_binning_mode = camera.getSpatialBinning()
        print("Current Spatial Binning mode is: ", current_binning_mode)
        camera.setSpatialBinning(spatial_binning_modes.BINNING_2)
        
        
    # For this to work, a classifier must be trained with blackstudio and must be transferred to the camera
    # also the camera mode must be the same as used for training the classifier
    if (EXAMPLE_ON_CAMERA_CLASSIFICATION_ENABLED and camera.ON_CAMERA_CLASSIFIER_SUPPORT):
        camera.setMode(0) # use mode used for training
        camera.setSpatialBinning(spatial_binning_modes.NO_BINNING)# set spatial binning as used for training the classifier
        camera.set_pixel_format(pixel_formats.CLASSES_COLOUR)
    
    
    if(EXAMPLE_DIFFERENTIAL_TRIGGER and camera.DIFFERENTIAL_TRIGGER_SUPPORT):
        print("Input Trigger Pin Mode is:", camera.getInputTriggerPinMode)
        print("Output Trigger Pin Mode is: ", camera.getOutputTriggerPinMode)

        
        camera.setInputTriggerPinMode(HAIP_BlackIndustry.DIFFERENTIAL)
        camera.setOutputTriggerPinMode(HAIP_BlackIndustry.DIFFERENTIAL)
        
        print("Set Input Trigger Pin Mode to: ", camera.getInputTriggerPinMode)
        print("Set Output Trigger Pin Mode to: ", camera.getOutputTriggerPinMode)
        

        
    
    
    print("Current Mode Maximum FPS:", camera.getCurrentMaxFPS())
    print("Maximum exposure when maximum FPS used:", camera.getCurrentMaxExposure())
    print("Current Mode Resolution:", camera.getCurrentResolution())

    if (EXAMPLE_BLOCK_TCP_UNTIL_WARMUP):
        camera.setTCPBlockSendout(1)


    # start camera capture mode
    camera.startCameraStream()

    if (EXAMPLE_BLOCK_TCP_UNTIL_WARMUP):
        # wait 3 seconds
        time.sleep(3)

        # mode 0 - the next images that are ready for tcp will be sent (normal mode)
        # mode 1 - BLOCKED
        # mode 2 - the next images which arrival time at cpu is later than timestamp of this command will be sent
        # camera.setTCPBlockSendout(0);
        camera.setTCPBlockSendout(2)


    # receive 10 images
    for i in range(10):

        # receive the last image
        image = camera.getImage()

        # when there is no new image, wait some time and try again
        while(image is None):
            time.sleep(0.001)
            image = camera.getImage()

        # image is a numpy array
        print(image.shape)

        # test if data is inside
        if EXAMPLE_ON_CAMERA_CLASSIFICATION_ENABLED:
            # if classification is done on camera, spectral data is analyzed for classifying
            # classification classes are represented by RGB pixels
            # therefore, the result is diffently shaped
            print(image[::100,:])
        else:
            print(image[::100, ::10])

        
    print("temperature:", camera.getTemperature())

    # stop camera capture mode
    camera.stopCameraStream()

if __name__ == '__main__':
    example()