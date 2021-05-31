# Service installation

Two services:
 - detection
 - ingestion
provided as systemd services.

### Installation
Explicit for `ingestion`, do the same for `detection`:

  1. Add `ingestion-cfg.json` (copy from `pvt/venezia` submodule) to this folder
  2. Add `ingestion.service` (copy from `tools/venezia/service_template.service` submodule) service file
  3. Link (hard link only) it to services system folder
  ```
  $ sudo ln $(pwd)/ingestion.service /etc/systemd/system/ingestion.service
  ```
  4. Reload daemons, start service, enable service
  ```
  $ sudo systemctl daemon-reload
  $ sudo systemctl start ingestion.service
  $ sudo systemctl enable ingestion.service
  ```