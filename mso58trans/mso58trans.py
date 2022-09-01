import string
import pyvisa
import numpy as np

# データだけ転送する
class MSO58transData:
    def __init__(self, visa_address):
        self.visa_address = visa_address
        self.rm = pyvisa.ResourceManager()
        with self.rm.open_resource(self.visa_address) as inst:
            print(inst.query('*IDN?'))

    def transfer2byte(self, channel):
        waveform , t= None, None
        with self.rm.open_resource(self.visa_address) as inst:
            assert f"CH{channel}" in inst.query("DATa:SOUrce:AVAILable?")
            fastState = inst.query("HORizontal:FASTframe:STATE?").strip()
            fastState = not (("OFF" in fastState) or (fastState == "0"))
            frameCount = int(inst.query("HORizontal:FASTframe:COUNt?")) if fastState else 1
            if fastState:
                acquiredFrameCount = int(inst.query("ACQUIRE:NUMFRAMESACQUIRED?"))
                assert 0 == int(inst.query("ACQuire:STATE?")), "please stop acquiring"
                assert acquiredFrameCount > 0
                frameCount == acquiredFrameCount
            else:
                assert int(inst.query("ACQuire:NUMACq?")) > 0
            inst.write("DATa:ENCdg RIBinary")
            inst.write("WFMOutpre:BYT_Nr 2")
            inst.write(f"DATa:SOUrce CH{channel}")
            inst.write("DATa:STARt 1")
            points = int(inst.query("HORizontal:RECOrdlength?"))
            duration = float(inst.query("HORizontal:ACQDURATION?"))
            sampleRate = float(inst.query("HORizontal:SAMPLERate?"))
            vOffset = float(inst.query("WFMOutpre:YZEro?"))
            vIncrement = float(inst.query("WFMOutpre:YMUlt?"))
            tpos = float(inst.query("HORizontal:POSition?")) / 100
            startTime = -tpos * duration
            endTime = (1 - tpos) * duration
            inst.write("DATa:STOP " + str(points))
            inst.write("CURVe?")
            buff = inst.read_raw()
            nw = int(buff[1:2].decode('ascii'))
            waveform = None
            if fastState:
                fcount = len(buff) // (2*points + nw + 3)
                frames = [np.frombuffer(buff[(3+nw) * (f+1) - 1 + 2*points*f : (3+nw) * (f+1) - 1 + 2*points*(f+1)], dtype=np.dtype('>i2')) for f in range(fcount)]
                waveform = np.vstack(frames)
            else:
                waveform = np.frombuffer(buff[2+nw:2+nw + points*2], dtype=(np.dtype('>i2')))
            waveform = waveform * vIncrement + vOffset
            t = np.linspace(startTime, endTime, points)
        return t, waveform

    def transfer1byte(self, channel):
        waveform , t= None, None
        with self.rm.open_resource(self.visa_address) as inst:
            assert f"CH{channel}" in inst.query("DATa:SOUrce:AVAILable?")
            fastState = inst.query("HORizontal:FASTframe:STATE?").strip()
            fastState = not (("OFF" in fastState) or (fastState == "0"))
            frameCount = int(inst.query("HORizontal:FASTframe:COUNt?")) if fastState else 1
            if fastState:
                acquiredFrameCount = int(inst.query("ACQUIRE:NUMFRAMESACQUIRED?"))
                assert 0 == int(inst.query("ACQuire:STATE?")), "please stop acquiring"
                assert acquiredFrameCount > 0
                frameCount == acquiredFrameCount
            else:
                assert int(inst.query("ACQuire:NUMACq?")) > 0
            inst.write("DATa:ENCdg RIBinary")
            inst.write("WFMOutpre:BYT_Nr 1")
            inst.write(f"DATa:SOUrce CH{channel}")
            inst.write("DATa:STARt 1")
            points = int(inst.query("HORizontal:RECOrdlength?"))
            duration = float(inst.query("HORizontal:ACQDURATION?"))
            sampleRate = float(inst.query("HORizontal:SAMPLERate?"))
            vOffset = float(inst.query("WFMOutpre:YZEro?"))
            vIncrement = float(inst.query("WFMOutpre:YMUlt?"))
            tpos = float(inst.query("HORizontal:POSition?")) / 100
            startTime = -tpos * duration
            endTime = (1 - tpos) * duration
            inst.write("DATa:STOP " + str(points))
            inst.write("CURVe?")
            buff = inst.read_raw()
            nw = int(buff[1:2].decode('ascii'))
            waveform = None
            if fastState:
                fcount = len(buff) // (1*points + nw + 3)
                frames = [np.frombuffer(buff[(3+nw) * (f+1) - 1 + 1*points*f : (3+nw) * (f+1) - 1 + 1*points*(f+1)], dtype=np.dtype('>i1')) for f in range(fcount)]
                waveform = np.vstack(frames)
            else:
                waveform = np.frombuffer(buff[2+nw:2+nw + points*1], dtype=(np.dtype('>i1')))
            waveform = waveform * vIncrement + vOffset
            t = np.linspace(startTime, endTime, points)
        return t, waveform



# WFMファイルを転送する。ゴリ押し。
class MSO58transWFM:
    def __init__(self, visa_address):
        self.visa_address = visa_address
        self.rm = pyvisa.ResourceManager()

        with self.rm.open_resource(self.visa_address) as inst:
            print(inst.query('*IDN?'))

    def _tempSaveChannel(self, i):
        assert i in [j+1 for j in range(8)]
        dest = f"C:/Users/Tek_Local_Admin/wfm/temp_ch{i}.wfm"
        with self.rm.open_resource(self.visa_address) as inst:
            inst.write(f'FILESystem:DELEte "{dest}"')
            inst.write(f'SAVe:WAVEform CH{i}, "{dest}"')

    def _transferWFM(self, i) -> bytes:
        assert i in [j+1 for j in range(8)]
        dest = f"C:/Users/Tek_Local_Admin/wfm/temp_ch{i}.wfm"
        buff = None
        dir = "C:/Users/Tek_Local_Admin/wfm/"
        with self.rm.open_resource(self.visa_address) as inst:
            inst.write(f'FILESystem:CWD "{dir}"')
            existance = f"temp_ch{i}.wfm" in inst.query('FILESystem:DIR?')
            if existance:
                inst.write(f'FILESYSTEM:READFILE "{dest}"')
                buff = inst.read_raw()
        return buff
    
    def transfer(self, channel):
        self._tempSaveChannel(channel)
        buff = self._transferWFM(channel)
        return buff

    def saveWFM(self, channel: int, path: string):
        buff = self.transfer(channel)
        assert buff != None, "Osilo returned nothing, confirm enabled channel."
        with open(path, "wb") as f:
            f.write(buff)
    

