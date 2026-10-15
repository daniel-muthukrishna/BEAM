# Tessbeam web tool

Estimate TESS stray light for any time or sector.

## Access

The server runs on **pdogpu6** and currently only accepts local users. From any machine with pdo access:

1. Open the tunnel from your own computer:

   ```bash
   ssh -N -L 7860:localhost:7860 <your-username>@pdogpu6
   ```

   - The terminal will look frozen. That's normal: it is holding the tunnel open.
   - Leave it running while you use the tool. Press `Ctrl+C` to close it.
   - If you normally reach pdogpu6 through a gateway machine, connect the same way you usually do.

2. Open the tool in your browser:

   **http://localhost:7860/**

> If port 7860 is already in use on your computer, change only the first number, e.g. `-L 7861:localhost:7860`, then visit `http://localhost:7861/`.

## Usage

### Inputs

| Input | Required | Description |
|---|---|---|
| Time **or** sector | Yes | A time (UTC ISO, e.g. `2019-05-09T05:15:00`, or TJD) or a TESS sector number (e.g. `11`). |
| Target RA/Dec | No | Marks the target on the image and adds a stray-light light curve at that position. |

### Outputs

- **Stray-light estimate** for the pointing at the requested time or sector.
- **Target light curve** (only if RA/Dec is given): predicted stray light at the target over the time range or sector.

### Downloads

All results can be downloaded directly from the page:

- PNG image
- Movie
- FITS cube
