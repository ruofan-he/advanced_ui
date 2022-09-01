import numpy as np
import scipy as sp
import scipy.signal # for Digital Filter
import scipy.fftpack
import matplotlib.pyplot as plt
import matplotlib.cm
import re
import os
import json
import visa

class Oscilloscope:
    def __init__(self):
        pass
    def getWaveform(self, ch):
        pass
    def isWaiting(self):
        pass
    def getConfig():
        pass
    def setConfig():
        pass

class WaveformCollection:
    def __init__(self, config_com):
        self.config_com = config_com
        self.waves = {}
    @staticmethod
    def load(dirname):
        with open(dirname + "/config.json", "r") as fp:
            config = json.load(fp)
        i = WaveformCollection(config["config_com"])
        for name, value in config["waves"].items():
            i.append(name, value[0], np.load(dirname + "/" + value[1]))
        return i
    def save(self, dirname):
        def toFilename(n):
            return re.sub(r'[\\|/|:|?|.|"|<|>|\|]', '_', n) + ".npy"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        waves_save = {}
        for name, value in self.waves.items():
            fname = toFilename(name)
            waves_save[name] = (value[0], fname)
            np.save(dirname + "/" + fname, value[1])
        with open(dirname + "/config.json", "w") as fp:
            json.dump({"config_com": self.config_com, "waves": waves_save}, fp)
    def append(self, name, config_ch, waveform):
        self.waves[name] = (config_ch, waveform)
    def names(self):
        return self.waves.keys()
    def time(self):
        return np.linspace(self.config_com["horizontal"]["start time"], self.config_com["horizontal"]["stop time"], self.config_com["horizontal"]["points"])
    def waveform_raw(self, name):
        return self.waves[name][1]
    def waveform_voltage(self, name):
        config_ch = self.config_ch(name)
        return self.waveform_raw(name) * config_ch["v increment"] + config_ch["offset"]
    def config_ch(self, name):
        return self.waves[name][0]
    def plot(self, legend_loc="best", outfile=None):
        plt.figure(figsize=(10,6))
        ax = plt.subplot(111)
        plt.grid(which='major',color='gray',linestyle='-')
        plt.grid(which='minor',color='gray',linestyle='-')
        ax.grid(True)
        ax.set_ylabel("voltage [V]")
        ax.set_xlabel("time [s]")
        time = self.time()
        cd = 1 / len(self.names())
        for i, name in enumerate(self.names()):
            c = matplotlib.cm.prism(i * cd)
            if self.config_com["com"]["segmented"]:
                first = True
                for w in self.waveform_voltage(name):
                    if first:
                        ax.plot([np.NaN], label=name, color=c, lw=1)
                        first = False
                    ax.plot(time, w, lw=1, color=c, alpha=1/self.config_com["com"]["segment count"])
            else:
                ax.plot(time, self.waveform_voltage(name), label=name, lw=1, color='c')
        if legend_loc != None:
            plt.legend(loc=legend_loc)
        if outfile != None:
            plt.savefig(outfile)
    # Additional methods for mode function by Takeda (2018/6/13)            
    def apply_modefunction(self, shape, gamma, center, width, spacing, number): # Apply mode function and convert waveforms to quadratures
        quadrature = {}
        time = self.time()
        for name in self.names():
            quadrature[name]=np.array([])
            if self.config_com["com"]["segmented"]:
                first = True
                for w in self.waveform_voltage(name):
                    for i in range(number):
                        if i==0:
                            quadrature_singleshot=np.dot(w, modefunction(shape,gamma,center+i*spacing,width,time))*self.config_com["horizontal"]["t increment"]
                        else:
                            quadrature_singleshot=np.append(quadrature_singleshot, np.dot(w, modefunction(shape,gamma,center+i*spacing,width,time))*self.config_com["horizontal"]["t increment"])
                    if first:
                        quadrature[name]=quadrature_singleshot
                    else:
                        quadrature[name]=np.vstack((quadrature[name],quadrature_singleshot))
                    first = False                        
            else:
                quadrature_singleshot = {}
                for i in range(number):
                    if i==0:
                        quadrature_singleshot=np.dot(self.waveform_voltage(name), modefunction(shape,gamma,center+i*spacing,width,time))*self.config_com["horizontal"]["t increment"]
                    else:
                        quadrature_singleshot=np.append(quadrature_singleshot, np.dot(self.waveform_voltage(name), modefunction(shape,gamma,center+i*spacing,width,time))*self.config_com["horizontal"]["t increment"])
                quadrature[name]=quadrature_singleshot
        return quadrature        
    def plot_modefunction(self, shape, gamma, center, width, spacing, number): # Plot multi mode function to check shape & position
        plt.figure(figsize=(10,6))
        ax = plt.subplot(111)
        plt.grid(which='major',color='gray',linestyle='-')
        plt.grid(which='minor',color='gray',linestyle='-')
        ax.grid(True)
        ax.set_ylabel("amplitude (arb. unit)")
        ax.set_xlabel("time [s]")
        time = self.time()
        for i in range(number):
            ax.plot(time, modefunction(shape,gamma,center+i*spacing,width,time), lw=1, color=matplotlib.cm.hsv(i/number))
    # Additional methods for FFT by Takeda (2018/6/14)
    def fft(self, legend_loc="best", outfile=None):
        plt.figure(figsize=(10,6))
        ax = plt.subplot(111)
        plt.grid(which='major',color='gray',linestyle='-')
        plt.grid(which='minor',color='gray',linestyle='-')
        ax.grid(True)
        ax.set_ylabel("Power [dB]")
        ax.set_xlabel("Frequency [Hz]")
        N=self.config_com["horizontal"]["points"]
        freq=np.linspace(0,1/self.config_com["horizontal"]["t increment"],N) # Create values for frequency axis
        np.save('freq.npy', freq)
        cd = 1 / len(self.names())
        hammingwindow=np.hamming(N) # Window function
        for i, name in enumerate(self.names()): # Perform FFT for each channel
            if self.config_com["com"]["segmented"]: # for FastFrame data, calculate FFT spectrum for each trace and take the average
                F_abs=np.abs(np.fft.fft(hammingwindow*self.waveform_voltage(name))) # apply window function and perform FFT
                F_abs_mean=np.mean(F_abs, axis=0)
                Power=10*np.log10(F_abs_mean**2)
                ax.plot(freq[:int(N/2)+1],Power[:int(N/2)+1], label=name, lw=1, color=matplotlib.cm.hsv(i * cd))
            else:
                F_abs=np.abs(np.fft.fft(hammingwindow*self.waveform_voltage(name))) # for signle-shot data, just calculate FFT spectrum
                Power=10*np.log10(F_abs**2)
                ax.plot(freq[:int(N/2)+1],Power[:int(N/2)+1], label=name, lw=1, color=matplotlib.cm.hsv(i * cd))
            np.save(str(name), Power)
        if legend_loc != None:
            plt.legend(loc=legend_loc)
        if outfile != None:
            plt.savefig(outfile)
    # Additional methods for digital filter by Takeda (2018/6/18)
    def filter(self,filter_type,cutoff,numtaps): # filter_type="LPF" or "HPF", cutoff[Hz]
        fs=1/self.config_com["horizontal"]["t increment"] # Sampling frequency
        fnyq=fs/2 # Nyquist frequency
        fc=cutoff/fnyq # Cutoff frequency normalized by Nyquist frequency
        if filter_type=="LPF":
            b=scipy.signal.firwin(numtaps,fc) # FIR LPF
        else:
            b=scipy.signal.firwin(numtaps,fc,pass_zero=False) # FIR HPF
        filtered_voltage = {}
        filtered_waveform=WaveformCollection(self.config_com)
        for name in self.names():
            filtered_voltage=np.array([])
            if self.config_com["com"]["segmented"]:
                first = True
                for w in self.waveform_voltage(name):
                    if first:
                        filtered_voltage=scipy.signal.lfilter(b,1,w) # Apply filter
                    else:
                        filtered_voltage=np.vstack((filtered_voltage,scipy.signal.lfilter(b,1,w)))  # Apply filter
                    first = False 
            else:
                filtered_voltage=scipy.signal.lfilter(b,1,self.waveform_voltage(name)) # Apply filter
            filtered_raw=(filtered_voltage-self.config_ch(name)["offset"])/self.config_ch(name)["v increment"] # Convert to raw data
            filtered_waveform.append(name,self.config_ch(name),filtered_raw)
        return filtered_waveform

# Definition of mode function by Takeda
def modefunction(shape, gamma, center, width, time):
    if shape == "Exponential":
        return np.exp(-gamma*np.absolute(time-center))*np.heaviside(width/2+time-center,1)*np.heaviside(width/2-time+center,1)
    elif shape == "Gaussian":
        return np.exp((-gamma**2)*((time-center)**2))*np.heaviside(width/2+time-center,1)*np.heaviside(width/2-time+center,1)
    elif shape == "DCreject":
        return np.exp((-gamma**2)*((time-center)**2))*(time-center)*np.heaviside(width/2+time-center,1)*np.heaviside(width/2-time+center,1)

class Oscilloscope_Tektronix(Oscilloscope):
    def __init__(self, visa_addr):
        rm = visa.ResourceManager()
        self.inst = rm.open_resource(visa_addr)
        self.idn = self.inst.query("*IDN?").strip()
        self.inst.write("ACQuire:INTERPEightbit OFF")
        self.inst.write("DATa:ENCdg RIBinary")
        self.inst.write("WFMOutpre:BYT_Nr 2")
        super().__init__()
    def setLargeChunk(self):
        self.inst.chunk_size = 10 * 1024**2
    def setSmallChunk(self):
        self.inst.chunk_size = 20 * 1024
    def loadConfig_com(self):
        config_com = {}
        tpos = float(self.inst.query("HORizontal:POSition?")) / 100
        trange = float(self.inst.query("HORizontal:ACQDURATION?"))
        config_com["com"] = {}
        config_com["horizontal"] = {}
        config_com["trigger"] = {}
        config_com["com"]["IDN"]                  = self.idn
        config_com["com"]["acq type"]             = self.inst.query("ACQuire:MODe?").strip()
        faststate                                 = self.inst.query("HORizontal:FASTframe:STATE?").strip()
        config_com["com"]["segmented"]            = not (("OFF" in faststate) or (faststate == "0"))
        config_com["com"]["segment count"]        = int(self.inst.query("HORizontal:FASTframe:COUNt?")) if config_com["com"]["segmented"] else 1
        config_com["horizontal"]["start time"]    = -tpos * trange
        config_com["horizontal"]["stop time"]     =  (1 - tpos) * trange
        config_com["horizontal"]["points"]        = int(self.inst.query("HORizontal:RECOrdlength?"))
        config_com["horizontal"]["t increment"]   = 1/float(self.inst.query("HORizontal:SAMPLERate?"))
        config_com["trigger"]["type"]             = self.inst.query("TRIGger:A:TYPe?").strip()
        config_com["trigger"]["source"]           = self.inst.query("TRIGger:A:EDGE:SOUrce?").strip() if config_com["trigger"]["type"] == "EDGE" else ""
        self.config_com = config_com
    def single(self):
        self.inst.write(":SINGle")
    def isFinished(self):
        return self.inst.query(":ASTate?").startswith("ADONE")
    def digitize(self):
        self.inst.write(":DIGitize")
    def setConfig():
        pass
    def getConfig_com(self):
        return self.config_com
    def getConfig_ch(self, ch):
        config_ch = {}
        self.inst.write("DATa:SOUrce CH" + str(ch))
        config_ch["offset"]                       = float(self.inst.query("WFMOutpre:YZEro?"))
        config_ch["range"]                        = float(self.inst.query("CH" + str(ch) + ":SCAle?")) * 10
        config_ch["v increment"]                  = float(self.inst.query("WFMOutpre:YMUlt?"))
        config_ch["input"]                        = self.inst.query("CH" + str(ch) + ":TERmination?").strip()
        return config_ch
    def getWaveform(self, ch):
        self.inst.write("DATa:SOUrce CH" + str(ch))
        self.inst.write("DATa:STARt 1")
        self.inst.write("DATa:STOP " + str(self.config_com["horizontal"]["points"]))
        self.inst.write("CURVe?")
        data = self.inst.read_raw()
        nw = int(data[1:2].decode('ascii'))
        points = self.config_com["horizontal"]["points"]
        if self.config_com["com"]["segmented"]:
            fcount = len(data) // (2*points + nw + 3)
            frames = (np.frombuffer(data[(3+nw) * (f+1) - 1 + 2*points*f : (3+nw) * (f+1) - 1 + 2*points*(f+1)], dtype=np.dtype('>i2')) for f in range(fcount))
            waveform = np.vstack(frames)
        else:
            waveform = np.frombuffer(data[2+nw:2+nw + points*2], dtype=(np.dtype('>i2')))
        return waveform
    def close(self):
        self.inst.close()

# Class to control Quantum Composer Delay Generator by Takeda (2018/6/19)
class QuantumComposer:
    def __init__(self, visa_addr):
        rm = visa.ResourceManager()
        self.inst = rm.open_resource(visa_addr)
    def setDelay(self, channel, delay): # channel: 1-8, delay [s]
        self.inst.write(":PULSE" + str(channel) + ":DELAY " + format(delay, '.9f'))
        #self.inst.write(":PULSE" + str(channel) + ":DELAY " + "{0:.9f}".format(delay))
        #self.inst.write(":PULSE" + str(channel) + ":DELAY " + str(delay))
    def setWidth(self, channel, width): # channel: 1-8, width [s]
        self.inst.write(":PULSE" + str(channel) + ":WIDT " + str(width))
