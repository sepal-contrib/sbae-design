## 2.0.0 (2026-08-20)

### Feat

- complete the Spanish and French catalogs
- translate the remaining app surfaces
- translate the app through a message catalog
- migrate to pysepal 4.0
- default map min_zoom of 5
- Made sure raster optim progress is visible when modal is closed.
- get rid off gee for Aoi calculations
- Included balanced allocation. Show EUA only for Neymans and set all classes as high by default
- move export to right side panel
- update logger

### Fix

- only borrow a generic port-forwarding prefix for vector tiles
- reject non-raster classification maps instead of crashing the map
- wait for the class palette before drawing the raster
- keep scratch dirs off NFS
- bind the PMTiles tile server to 127.0.0.1 (#17)
- put the interpreter's bin on PATH so tippecanoe resolves under SEPAL (#14)
- add pythonpath to pytest config for CI compatibility
- configure PROJ_DATA at startup to avoid CRS errors
- fix aoi import

### Refactor

- remove debugging sleep
- Made sure seed can be always seen by user
- deleted unused file
- reorganize project into component-based structure with proper imports
