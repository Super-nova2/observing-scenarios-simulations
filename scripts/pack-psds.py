#!/usr/bin/env python
"""Pack ASCII injection files into a psd.xml file."""

import os
from argparse import SUPPRESS, FileType

import lal
import lal.series
import numpy as np
from igwn_ligolw.utils import SignalsTrap, write_fileobj
from ligo.skymap.tool import ArgumentParser, register_to_xmldoc

# Command line interface
detector_names = [d.frDetector.prefix for d in lal.CachedDetectors]
detector_long_names = [d.frDetector.name for d in lal.CachedDetectors]
parser = ArgumentParser()
parser.add_argument(
    "-o",
    "--output",
    metavar="OUT.xml[.gz]",
    type=FileType("wb"),
    default="-",
    help="Name of output file [default: stdout]",
)
for name, long_name in zip(detector_names, detector_long_names):
    parser.add_argument(
        f"--{name}",
        metavar="SPECTRUM.txt",
        type=FileType("r"),
        default=SUPPRESS,
        help=f"ASD filename for {long_name} detector unless --{name}-is-psd is set",
    )
    parser.add_argument(
        f"--{name}-column",
        metavar="COLUMN",
        default=SUPPRESS,
        help=f"Column name for {long_name} detector",
    )
    parser.add_argument(
        f"--{name}-is-psd",
        action="store_true",
        default=SUPPRESS,
        help=f"Treat the {long_name} detector input as PSD instead of ASD",
    )
args = parser.parse_args()

psds = {}
for name in detector_names:
    psd_file = getattr(args, name, None)
    if psd_file is None:
        continue

    column = getattr(args, f"{name}_column", None)
    if column is None:
        f, spectrum = np.loadtxt(psd_file).T
    else:
        data = np.genfromtxt(psd_file, names=True)
        f = data[data.dtype.names[0]]
        spectrum = data[column]
    if getattr(args, f"{name}_is_psd", False):
        psd = spectrum
    else:
        psd = np.square(spectrum)

    f0 = 10.0
    fmax = 4096.0
    df = 1.0

    fgrid = np.arange(f0, fmax, df)
    series = lal.CreateREAL8FrequencySeries(
        (psd_file.name).split(".")[0], 0, f0, df, lal.SecondUnit, len(fgrid)
    )
    series.data.data = np.exp(np.interp(np.log(fgrid), np.log(f), np.log(psd)))

    psds[name] = series

xmldoc = lal.series.make_psd_xmldoc(psds, encoding="base64")
register_to_xmldoc(xmldoc, parser, args)

with SignalsTrap():
    write_fileobj(
        xmldoc,
        args.output,
        compress="gz" if os.path.splitext(args.output.name)[-1] == ".gz" else False,
    )
