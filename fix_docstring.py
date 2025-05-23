#!/usr/bin/env python3
with open("component/scripts/map_utils.py", "r") as f:
    content = f.read()

# Replace the docstring with the fixed version
old_docstring = '''    """
    Adds Raster (ImageOverlay/DataURL) or Vector (GeoJSON) layer.
    Returns the created layer object or None if unsuccessful.
    """'''

new_docstring = '''    """Adds Raster (ImageOverlay/DataURL) or Vector (GeoJSON) layer.

    Returns the created layer object or None if unsuccessful.
    """'''

fixed_content = content.replace(old_docstring, new_docstring)

with open("component/scripts/map_utils.py", "w") as f:
    f.write(fixed_content)

print("Docstring fixed!")
