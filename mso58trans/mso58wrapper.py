import pyvisa
from .mso58trans import MSO58transData


class MSO58Wrapper(MSO58transData):
    def push_single(self):
        with self.rm.open_resource(self.visa_address) as inst:
            inst.write('ACQUIRE:STOPAFTER SEQUENCE')
            inst.write('ACQUIRE:STATE ON')
