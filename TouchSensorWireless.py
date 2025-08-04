import numpy as np
import json5
import json
from GenericReceiver import GenericReceiverClass
import socket
import select
import threading
from typing import List
from Sensor import Sensor
import aioconsole
import serial
import webbrowser
import re
import time
import asyncio
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import serial_asyncio
from flaskApp.index import update_sensors, replay_sensors, start_server, notify_device_connected
import utils
import os
import platform
import qrcode
import utils  # assuming this contains your getUnixTimestamp function
import cv2
from pathlib import Path


class SerialProtocol(asyncio.Protocol):
    def __init__(self, receiver):
        """Initialize protocol with a reference to SerialReceiver."""
        self.receiver = receiver
        self.transport = None

    def connection_made(self, transport):
        """Called when the serial connection is established."""
        self.transport = transport
        print("Connection established.")

    def data_received(self, data):
        """Called when new data is received."""
        self.receiver.buffer.put_nowait(data)  # Directly put data without creating a task

    def connection_lost(self, exc):
        """Called when the connection is lost or closed."""
        print("Connection lost.")


class WifiReceiver(GenericReceiverClass):
    def __init__(self,numNodes,sensors:List[Sensor], tcp_ip="10.0.0.67", tcp_port=7000, record=True, stopFlag=None):
        super().__init__(numNodes,sensors,record)
        self.TCP_IP = tcp_ip
        self.tcp_port = tcp_port
        self.connection_is_open = False
        self.connections = {}
        self.setup_TCP()
        self.stopFlag = stopFlag
    
    def setup_TCP(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.TCP_IP, self.tcp_port))
        sock.listen(len(self.sensors))  # Listen for N connections
        print("Waiting for connections")
        while len(self.connections) < len(self.sensors):
            connection, client_address = sock.accept()
            print("Connection found")
            sensorId = self.getSensorIdFromBuffer(connection)
            print(f"Connection found from {sensorId}")
            notify_device_connected(sensorId,True)
            self.connections[sensorId]=connection
        sock.settimeout(30)
        print("All connections found")

    def getSensorIdFromBuffer(self, connection):
        while True:
            ready_to_read, ready_to_write, in_error = select.select([connection], [], [], 30)
            if len(ready_to_read)>0:
                numBytes = 1+(int(self.numNodes)+1)*2+4
                inBuffer =   connection.recv(numBytes, socket.MSG_PEEK)
                if len(inBuffer) >= numBytes:
                    sendId, startIdx, sensorReadings, packet = self.unpackBytesPacket(inBuffer)
                return sendId

    def reconnect(self, sensorId):
        print(f"Reconnecting to sensor {sensorId}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.TCP_IP, self.tcp_port))
        sock.listen(len(self.sensors))
        print("waiting for connections")
        found_Conn = False
        while not found_Conn:
            connection, client_address = sock.accept()
            connectionSensorId = self.getSensorIdFromBuffer(connection)
            if connectionSensorId == sensorId:
                print(f"Connection from Sensor {sensorId} found")
                self.connections[sensorId]=connection
                found_Conn = True
            else:
                print(f"Connection refused from {client_address}")
                connection.close()

    async def receiveData(self, sensorId):
        print("Receiving Data")
        while not self.stopFlag.is_set():
            connection = self.connections[sensorId]
            ready_to_read, ready_to_write, in_error = await asyncio.get_event_loop().run_in_executor(
                None, select.select, [connection], [], [], 30)
            if len(ready_to_read)>0:
                numBytes = 1+(int(self.numNodes)+1)*2+4
                inBuffer =   await asyncio.get_event_loop().run_in_executor(None, connection.recv, numBytes, socket.MSG_PEEK)
                if len(inBuffer) >= numBytes:
                    data = await asyncio.get_event_loop().run_in_executor(None, connection.recv, numBytes)
                    sendId, startIdx, sensorReadings, packet = self.unpackBytesPacket(data)
                    sensor = self.sensors[sendId]
                    if(sensor.intermittent):
                        sensor.processRowIntermittent(startIdx,sensorReadings,packet,record=self.record)
                    else:
                        if (startIdx==20000):
                            sensor.processRowReadNode(sensorReadings,packet,record=self.record)
                        else:
                            sensor.processRow(startIdx,sensorReadings,packet,record=self.record)
            else:
                print(f"Sensor {sensorId} is disconnected: Reconnecting...")
                await asyncio.get_event_loop().run_in_executor(None, connection.shutdown, 2)
                await asyncio.get_event_loop().run_in_executor(None, connection.close)
                self.reconnect(sensorId)

    def startReceiverThreads(self):
        tasks = []
        for sensorId in self.connections:
            task = self.receiveData(sensorId)
            tasks.append(task)
        return tasks


class BLEReceiver(GenericReceiverClass):
    def __init__(self, numNodes, sensors: List[Sensor],record=True, stopFlag=None):
        super().__init__(numNodes, sensors, record)
        self.deviceNames = [(sensor.deviceName, sensor.id) for sensor in sensors]
        self.clients={}
        self.stopFlag = stopFlag

    async def connect_to_device(self, lock, deviceTuple):
        def on_disconnect(client):
            print(f"Device {deviceTuple[0]} disconnected")
            # asyncio.create_task(self.connect_to_device(lock, deviceName))
        async with lock:
            device = await BleakScanner.find_device_by_name(deviceTuple[0],timeout=30)
            if device:
                print(f"Found device: {deviceTuple[0]}")
                client = BleakClient(device)
                client.set_disconnected_callback(on_disconnect)
                self.clients[deviceTuple[0]] = client
                await client.connect()

        def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray):
                sendId, startIdx, sensorReadings, packet = self.unpackBytesPacket(data)
                sensor = self.sensors[sendId]
                if(sensor.intermittent):
                    sensor.processRowIntermittent(startIdx,sensorReadings,packet,record=self.record)
                else:
                    if (startIdx==20000):
                        sensor.processRowReadNode(sensorReadings,packet,record=self.record)
                    else:
                        sensor.processRow(startIdx,sensorReadings,packet,record=self.record)

        try:
            await client.start_notify("1766324e-8b30-4d23-bff2-e5209c3d986f", notification_handler)
            notify_device_connected(deviceTuple[1], True)
            print(f"Connected to {deviceTuple[0]}")
        except Exception as e:
            notify_device_connected(deviceTuple[1], False)
            print(f"Failed to connect to {deviceTuple[0]}: {e}")
    
    import asyncio

    async def stopReceiver(self, max_retries=30, delay=2):
        for deviceName, client in self.clients.items():
            retries = 0
            while retries < max_retries:
                try:
                    if client.is_connected:
                        await client.stop_notify("1766324e-8b30-4d23-bff2-e5209c3d986f")
                        # await client.disconnect()
                        print(f"{deviceName}: Notifications stopped and client disconnected.")
                    else:
                        print(f"{deviceName}: Client already disconnected.")
                    break  # Exit retry loop if successful
                except Exception as e:
                    retries += 1
                    print(f"Error stopping client {deviceName} (attempt {retries}/{max_retries}): {e}")
                    await asyncio.sleep(delay)
            else:
                print(f"{deviceName}: Failed to disconnect after {max_retries} retries.")
        print("All clients cleaned up.")


    def startReceiverThreads(self):
        lock = asyncio.Lock()
        tasks = [self.connect_to_device(lock, name) for name in self.deviceNames]
        return tasks


class SerialReceiver(GenericReceiverClass):
    def __init__(self, numNodes, sensors, baudrate, stopFlag=None, record =True):
        super().__init__(numNodes, sensors, record)
        self.baudrate = baudrate
        self.stop_capture_event = False
        self.reader = None
        self.stopStr = bytes('wr','utf-8')
        self.stopFlag = stopFlag
        self.transports = []


    async def read_serial(self):
        serObjs = []
        print("Starting read task")
        for sensor in self.sensors:
            # ser = serial.Serial()
            port = self.sensors[sensor].port
            # print(ser.port)
            # ser.baudrate = self.baudrate
            # print(ser.baudrate)
            # ser.dtr = False
            # ser.rts = False
            # ser.timeout=1
            # ser.open()
            # serObjs.append(ser)
            loop = asyncio.get_running_loop()
            # transport, protocol = await serial_asyncio.connection_for_serial(loop, lambda: SerialProtocol(self), ser)
            print(f"making serial connection{self.sensors[sensor].id}")
            transport, protocol = await serial_asyncio.create_serial_connection(
                loop,
                lambda: SerialProtocol(self),
                port,
                baudrate=self.baudrate
            )
            notify_device_connected(self.sensors[sensor].id, True)
            self.transports.append(transport)
        
        
        # Keep running until stopFlag is set
        while not self.stopFlag.is_set():
            await asyncio.sleep(0.1)  # Avoid blocking loop

        print("Stopping serial reader.")
        

    

    async def stopReceiver(self):
        for transport in self.transports:
            transport.close()
        print("Transports stopped")


    def startReceiverThreads(self):
        tasks=[]
        tasks.append(self.read_serial())
        tasks.append(self.read_lines())
        tasks.append(self.listen_for_stop())
        return tasks


def readConfigFile(file):
    with open(file, 'r') as file:
        data = json5.load(file)
    return data

def recordingsDirectory(recordings):
    if not os.path.exists(recordings):
        os.makedirs(recordings)

class MultiProtocolReceiver():
    def __init__(self, folder, configFilePath="./WiSensConfigClean.json"):
        self.recording_dir = os.path.join("ariarecordings", folder)
        if os.path.exists(self.recording_dir):
            raise FileExistsError(f"Directory '{self.recording_dir}' already exists.")
        else:
            os.makedirs(self.recording_dir)
            print(f"✅ Created directory: {self.recording_dir}")
        self.config = readConfigFile(configFilePath)
        self.sensors = self.config['sensors']
        self.bleSensors = []
        self.wifiSensors = []
        self.serialSensors = []
        self.allSensors = []
        self.stopFlag = asyncio.Event()
        for sensorConfig in self.sensors:
            sensorKeys = list(sensorConfig.keys())
            intermittent = False
            p = 15

            if 'intermittent' in sensorKeys:
                intermittent = sensorConfig['intermittent']['enabled']
                p = sensorConfig['intermittent']['p']

            deviceName = "Esp1"
            userNumNodes = 120

            match sensorConfig['protocol']:
                case 'wifi':
                    userNumNodes = self.config['wifiOptions']['numNodes']
                case 'ble':
                    deviceName = sensorConfig['deviceName']
                    userNumNodes = self.config['bleOptions']['numNodes']
                case 'serial':
                    userNumNodes = self.config['serialOptions']['numNodes']

            print("Hello 2s")
            numGroundWires = sensorConfig['endCoord'][1] - sensorConfig['startCoord'][1] + 1
            numReadWires = sensorConfig['endCoord'][0] - sensorConfig['startCoord'][0] + 1
            numNodes = min(userNumNodes,min(120, numGroundWires*numReadWires))
            id = sensorConfig['id']
            if id == 1:
                fileName = os.path.join(self.recording_dir,"leftPressure.hdf5")
            else:
                fileName = os.path.join(self.recording_dir,"rightPressure.hdf5")
            newSensor = Sensor(numGroundWires,numReadWires,numNodes,sensorConfig['id'],deviceName=deviceName,intermittent=intermittent, p=p, port=sensorConfig["serialPort"], fileName=fileName)
            
            match sensorConfig['protocol']:
                case 'wifi':
                    self.wifiSensors.append(newSensor)
                case 'ble':
                    self.bleSensors.append(newSensor)
                case 'serial':
                    self.serialSensors.append(newSensor)
            self.allSensors.append(newSensor)

        self.receivers = []
        self.receiveTasks = []
        self.activeThreads = []

    def updateConfig(self, config):
        self.config = config
        self.sensors = self.config['sensors']
        self.bleSensors = []
        self.wifiSensors = []
        self.serialSensors = []
        self.allSensors = []
        self.stopFlag = asyncio.Event()
        for sensorConfig in self.sensors:
            sensorKeys = list(sensorConfig.keys())
            intermittent = False
            p = 15

            if 'intermittent' in sensorKeys:
                intermittent = sensorConfig['intermittent']['enabled']
                p = sensorConfig['intermittent']['p']

            deviceName = "Esp1"
            userNumNodes = 120

            match sensorConfig['protocol']:
                case 'wifi':
                    userNumNodes = int(self.config['wifiOptions']['numNodes'])
                case 'ble':
                    deviceName = sensorConfig['deviceName']
                    userNumNodes = int(self.config['bleOptions']['numNodes'])
                case 'serial':
                    userNumNodes = int(self.config['serialOptions']['numNodes'])

            numGroundWires = sensorConfig['endCoord'][1] - sensorConfig['startCoord'][1] + 1
            numReadWires = sensorConfig['endCoord'][0] - sensorConfig['startCoord'][0] + 1
            numNodes = min(userNumNodes,min(120, numGroundWires*numReadWires))
            newSensor = Sensor(numGroundWires,numReadWires,numNodes,sensorConfig['id'],deviceName=deviceName,intermittent=intermittent, p=p, port=sensorConfig["serialPort"])
            
            match sensorConfig['protocol']:
                case 'wifi':
                    self.wifiSensors.append(newSensor)
                case 'ble':
                    self.bleSensors.append(newSensor)
                case 'serial':
                    self.serialSensors.append(newSensor)
            self.allSensors.append(newSensor)

        self.receivers = []
        self.receiveTasks = []
        self.activeThreads = []

    def programSensor(self, sensor_id, config):
        data = config
        # Find the sensor with the given ID
        sensor = next((s for s in data['sensors'] if s['id'] == sensor_id), None)
        if not sensor:
            raise ValueError(f"Sensor with ID {sensor_id} not found.")

        # Determine the protocol
        protocol = sensor.get('protocol')
        protocol_key = f"{protocol}Options"
        if protocol_key not in data:
            raise ValueError(f"Protocol '{protocol}' not supported.")

        # Get the protocol options
        protocol_options = data.get(protocol_key, {})

        #get readout options
        readout_options = data.get("readoutOptions",{})

        # Merge sensor data with the protocol options
        merged_data = {**sensor, **protocol_options, **readout_options}

        merged_data['resistance'] = max(0, min(127, (127 - int(merged_data['resistance']))))

        # Convert the merged data to a JSON string with proper quoting
        json_string = json.dumps(merged_data)
        print(json_string)

        if platform.system() =="Darwin":
            ser2 = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
            if "serialPort" in sensor:
                ser2.port=sensor["serialPort"]
                
            else:
                ser2.port=data['serialOptions']['port']

            print(ser2.port)
            print(ser2.baudrate)
            ser2.open()
        else:
            # Send the JSON string over the serial port
            ser = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
            if "serialPort" in sensor:
                ser.port=sensor["serialPort"]
            else:
                ser.port=data['serialOptions']['port']
            ser.open()
            ser.dtr = False  # Explicitly clear settings
            ser.rts = False
            ser.flush()
            ser.close()
            time.sleep(1)
            ser2 = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
            if "serialPort" in sensor:
                ser2.port=sensor["serialPort"]
                
            else:
                ser2.port=data['serialOptions']['port']

            print(ser2.port)
            print(ser2.baudrate)
            ser2.dtr = False  
            ser2.rts = False
            ser2.open()
            ser2.flush()

        ser2.write((json_string).encode('utf-8'))
        time.sleep(3)
        ser2.close()
        print("finished program")
        
    def calibrateSensor(self, sensor_id):
        data = self.config
        # Find the sensor with the given ID
        sensor = next((s for s in data['sensors'] if s['id'] == sensor_id), None)
        # Send the JSON string over the serial port
        ser = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
        if "serialPort" in sensor:
            ser.port=sensor["serialPort"]
        else:
            ser.port=data['serialOptions']['port']
        ser.dtr = False
        ser.rts = False
        try:
            ser.open()
            ser.write(b"calibrate\n")  # Send command (ensure correct encoding)

            # Wait for response
            while True:
                line = ser.readline().decode('utf-8').strip()  # Read line from serial
                match = re.search(r"Resistance\s*:\s*([\d.]+)", line)  # Match pattern
                if match:
                    resistance_value = float(match.group(1))  # Extract and convert value
                    return 127-resistance_value  # Return calibration value
        except Exception as e:
            print(f"Error: {e}")
        finally:
            ser.close()
    
    async def startReceiversAsync(self):
        await asyncio.gather(*self.receiveTasks)
        # await self.listen_for_stop()

    async def listen_for_stop(self):
        print("Listening for stop")
        while not self.stopFlag.is_set():
            input_str = await aioconsole.ainput("Press Enter to stop...\n")
            if input_str == "":
                print("Stop flag set")
                self.stopFlag.set()
        for receiver in self.receivers:
            if isinstance(receiver, BLEReceiver):
                await receiver.stopReceiver()


    async def listen_for_stop_web(self):
        print("Listening for stop")
        while not self.stopFlag.is_set():
            await asyncio.sleep(0.5)
        for receiver in self.receivers:
            if isinstance(receiver, BLEReceiver) or isinstance(receiver, SerialReceiver):
                await receiver.stopReceiver()


        self.receiveTasks = []  # Clear receiveTasks to allow fresh tasks on restart
        self.receivers = []  # Clear receivers list
        self.stopFlag.clear()  # Reset flag for future use
        print("All receivers stopped and cleaned up.")



    def initializeReceivers(self, record, web=False):
        if len(self.bleSensors)!=0:
            bleReceiver = BLEReceiver(int(self.config['bleOptions']['numNodes']),self.bleSensors, record)
            self.receivers.append(bleReceiver)
            self.receiveTasks += bleReceiver.startReceiverThreads()
        if len(self.wifiSensors)!=0:
            wifiReceiver = WifiReceiver(int(self.config['wifiOptions']['numNodes']),self.wifiSensors,self.config['wifiOptions']['tcp_ip'],int(self.config['wifiOptions']['port']), stopFlag=self.stopFlag, record=record)
            self.receivers.append(wifiReceiver)
            self.receiveTasks += wifiReceiver.startReceiverThreads()
        if len(self.serialSensors)!=0:
            serialReceiver = SerialReceiver(int(self.config['serialOptions']['numNodes']),self.serialSensors,self.config['serialOptions']['baudrate'],stopFlag=self.stopFlag,record=record)
            self.receivers.append(serialReceiver)
            self.receiveTasks += serialReceiver.startReceiverThreads()
        if web:
            self.receiveTasks.append(self.listen_for_stop_web())
        else:
            self.receiveTasks.append(self.listen_for_stop())

    def startReceiverThread(self):
        asyncio.run(self.startReceiversAsync())


    def timeSyncQR(self):
        ts = utils.getUnixTimestamp()
        display_ts = time.time()

        payload = {
            "gen": ts,
            "displayed": round(display_ts, 3)
        }

        qr_string = f"{payload}"

        # Generate QR code as PIL image
        qr_img = qrcode.make(qr_string)

        # Convert PIL image to OpenCV format
        qr_img = qr_img.convert("RGB")
        qr_np = np.array(qr_img)
        qr_cv = cv2.cvtColor(qr_np, cv2.COLOR_RGB2BGR)

        # Display using OpenCV (faster than external viewer)
        window_name = "Time Sync QR"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600, 600)
        cv2.imshow(window_name, qr_cv)

        # Wait for any key (or keep visible for N seconds)
        print("Press any key to close QR display...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def record(self):
        self.initializeReceivers(True)
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
        # captureThread.join()
        self.timeSyncQR()

    def visualizeAndRecord(self):
        self.initializeReceivers(True)
        threads=[]
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
        threads.append(captureThread)
        vizThread = threading.Thread(target=update_sensors, args=(self.allSensors,))
        vizThread.start()
        threads.append(vizThread)
        utils.start_nextjs()
        url = "http://localhost:3000"
        webbrowser.open_new_tab(url)
        start_server()
        for thread in threads:
            thread.join()


    def visualize(self):
        self.initializeReceivers(False)
        threads=[]
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
        threads.append(captureThread)
        vizThread = threading.Thread(target=update_sensors, args=(self.allSensors,))
        vizThread.start()
        threads.append(vizThread)
        # utils.start_nextjs()
        # url = "http://localhost:3000"
        # webbrowser.open_new_tab(url)
        start_server()
        for thread in threads:
            thread.join()

    def visualize_web(self):
        self.initializeReceivers(True, True)
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
        # self.activeThreads.append(captureThread)
        vizThread = threading.Thread(target=update_sensors, args=(self.allSensors,))
        vizThread.start()
        # self.activeThreads.append(vizThread)

    def replayData(self,fileDict, startTs=None,endTs=None, speed=1):
        pressureDict = {}
        totalFrames = None
        frameRate = None
        for sensorId in fileDict:
            pressure, fc, ts = utils.tactile_reading(fileDict[sensorId])
            startIdx = 0
            beginTs = ts[0]
            if startTs is not None:
                startIdx,beginTs = utils.find_closest_index(ts,startTs)
            endIdx, lastTs = len(ts), ts[-1]
            if endTs is not None:
                endIdx, lastTs = utils.find_closest_index(ts, endTs)
            if totalFrames is None or endIdx-startIdx<totalFrames:
                totalFrames = endIdx-startIdx
                frameRate = (totalFrames/(lastTs-beginTs)) * speed
            pressureDict[sensorId] = pressure[startIdx:endIdx,:,:]
        vizThread = threading.Thread(target=replay_sensors, args=(pressureDict,frameRate,totalFrames,))
        vizThread.start()
        utils.start_nextjs()
        url = "http://localhost:3000"
        # webbrowser.open_new_tab(url)
        start_server()

    def replayDataWeb(self,fileDict, startTs=None,endTs=None, speed=1):
        self.stopFlag.clear()
        pressureDict = {}
        totalFrames = None
        frameRate = None
        recordings_dir = Path("recordings")
        for sensorId in fileDict:
            print(sensorId,fileDict)
            pressure, fc, ts = utils.tactile_reading(recordings_dir / fileDict[sensorId])
            startIdx = 0
            beginTs = ts[0]
            if startTs is not None:
                startIdx,beginTs = utils.find_closest_index(ts,startTs)
            endIdx, lastTs = len(ts), ts[-1]
            if endTs is not None:
                endIdx, lastTs = utils.find_closest_index(ts, endTs)
            if totalFrames is None or endIdx-startIdx<totalFrames:
                totalFrames = endIdx-startIdx
                frameRate = (totalFrames/(lastTs-beginTs)) * speed
            pressureDict[sensorId] = pressure[startIdx:endIdx,:,:]
        vizThread = threading.Thread(target=replay_sensors, args=(pressureDict,frameRate,totalFrames,))
        vizThread.start()
        self.activeThreads.append(vizThread)

    # Sends all sensors (with real time pressure updates) as input to the custom method
    def runCustomMethod(self, method, record=False, viz=False):
        self.initializeReceivers(record)
        threads=[]
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
        threads.append(captureThread)
        customThread = threading.Thread(target=method, args=(self.allSensors,))
        customThread.start()
        threads.append(customThread)
        if viz:
            vizThread = threading.Thread(target=update_sensors, args=(self.allSensors,))
            vizThread.start()
            threads.append(vizThread)
            utils.start_nextjs()
            url = "http://localhost:3000"
            webbrowser.open_new_tab(url)
            start_server()
        for thread in threads:
            thread.join()
        

    

if __name__ == "__main__":
    utils.programSensor(1)
    myReceiver = MultiProtocolReceiver()
    myReceiver.visualize()
    

