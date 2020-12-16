import os
import sys
import json
import time

import win32service
import win32serviceutil
import win32event
import servicemanager

WORKSPACE = 'C:/Users/Alessandro/Codice'

try:
  sys.path.append(os.path.join(WORKSPACE, 'slides', 'tools', 'sybenik'))
  from ingestion_engine import ingestion
except Exception as e:
  raise Exception(f'ingestion svc : lib init failed {e}')

class PySvc(win32serviceutil.ServiceFramework):
  _svc_name_ = "sybenik-ingestion-svc"
  _svc_display_name_ = "Sybenik Cam Ingestion SVC"
  _svc_description_ = "Sybenik Camera Data Ingestion Service"

  def __init__(self, args):
    win32serviceutil.ServiceFramework.__init__(self,args)
    self._svc_config_file = os.path.join(WORKSPACE, 'slides', 'work_sybenik', 'ingestion-cfg.json')
    self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)

    with open(self._svc_config_file) as cin:
      config = json.load(cin)

    self._svc_clock_dt = 1
    self.ing = ingestion(configfile=self._svc_config_file)


  def SvcDoRun(self):
    rc = None
    while rc != win32event.WAIT_OBJECT_0:
      self.ing.do_task()
      rc = win32event.WaitForSingleObject(self.hWaitStop, 300)


  def SvcStop(self):
    self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
    win32event.SetEvent(self.hWaitStop)

if __name__ == '__main__':
  win32serviceutil.HandleCommandLine(PySvc)
