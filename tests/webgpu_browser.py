"""Generate CPU references and a browser test page beside an Emscripten build.

Usage: python tests/webgpu_browser.py --native-build build-upstream --web-dir build-web-upstream/web
Serve the web directory, open regression.html, and click Run browser regression.
The native build and its dependencies must be available on PATH.
"""
import argparse
from array import array
import json
from pathlib import Path
import subprocess
import sys


def floats(data):
    values = array('f', data)
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-build', type=Path, required=True)
    parser.add_argument('--web-dir', type=Path, required=True)
    args = parser.parse_args()
    destination = args.web_dir.resolve()
    data_dir = destination / 'regression-data'
    data_dir.mkdir(parents=True, exist_ok=True)
    executable = args.native_build.resolve() / ('oidnDenoise.exe' if sys.platform == 'win32' else 'oidnDenoise')
    cases = []
    for name, width, height, hdr, srgb, memory in [
        ('odd-srgb', 333, 197, False, True, -1),
        ('odd-hdr', 333, 197, True, False, -1),
        ('tiled-hdr', 1024, 1024, True, False, 64),
    ]:
        pixels = floats(
            (((x // 8)*17 + (y // 8)*29 + c*53) % 251) / 251 * (4 if hdr else 1)
            for y in range(height) for x in range(width) for c in range(3))
        (data_dir / (name+'.bin')).write_bytes(pixels)
        row_bytes = width * 3 * 4
        input_path = data_dir / (name+'.pfm')
        input_path.write_bytes(f'PF\n{width} {height}\n-1.0\n'.encode() + b''.join(
            pixels[y*row_bytes:(y+1)*row_bytes] for y in reversed(range(height))))
        output_path = data_dir / (name+'-cpu.pfm')
        command = [str(executable), '-d', 'cpu', '--hdr' if hdr else '--ldr', str(input_path),
                   '-o', str(output_path), '-q', 'high']
        if srgb:
            command.append('--srgb')
        if memory >= 0:
            command += ['--maxmem', str(memory)]
        subprocess.run(command, check=True, timeout=60)
        with output_path.open('rb') as reference:
            assert reference.readline().strip() == b'PF'
            assert reference.readline().strip() == f'{width} {height}'.encode()
            assert float(reference.readline()) < 0
            result = reference.read()
        (data_dir / (name+'-cpu.bin')).write_bytes(b''.join(
            result[y*row_bytes:(y+1)*row_bytes] for y in reversed(range(height))))
        cases.append(dict(name=name, width=width, height=height, hdr=hdr, srgb=srgb,
                          memory=memory, weights='rt_hdr.tza' if hdr else 'rt_ldr.tza'))
    (data_dir / 'cases.json').write_text(json.dumps(cases), encoding='utf-8')
    source = Path(__file__).with_name('webgpu_browser.html')
    (destination / 'regression.html').write_bytes(source.read_bytes())
    print(f'Open {destination / "regression.html"} through the local HTTP server.')


if __name__ == '__main__':
    main()
