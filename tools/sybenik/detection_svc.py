import os
import sys
import json
import time

import win32service
import win32serviceutil
import win32event
import servicemanager

try:
  sys.path.append(os.path.join(os.environ['SYBENIK_WORKSPACE'], 'tools', 'sybenik'))
  from detection_engine import detector
except Exception as e:
  raise Exception(f'detection svc : lib init failed {e}')

class PySvc(win32serviceutil.ServiceFramework):
  _svc_name_ = "sybenik-detection-svc"
  _svc_display_name_ = "Sybenik Cam Detection SVC"
  _svc_description_ = "Sybenik Camera Detection Service"

  def __init__(self, args):
    win32serviceutil.ServiceFramework.__init__(self,args)
    self._svc_config_file = os.path.join(os.environ['SYBENIK_WORKSPACE'], 'tools', 'sybenik', 'data', 'detection-cfg.json')
    self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)

    with open(self._svc_config_file) as cin:
      config = json.load(cin)

    self._svc_clock_dt = 1
    self.det = detector(configfile=self._svc_config_file)


  def SvcDoRun(self):
    rc = None
    while rc != win32event.WAIT_OBJECT_0:
      self.det.do_task()
      rc = win32event.WaitForSingleObject(self.hWaitStop, 300)


  def SvcStop(self):
    self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
    win32event.SetEvent(self.hWaitStop)

if __name__ == '__main__':
  win32serviceutil.HandleCommandLine(PySvc)
