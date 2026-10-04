# QE phonon file workflow

The command scripts in this directory are thin wrappers around the installed
`pyeph.preprocessing.qe` package. Use the complete workflow and qualification
guide in [`docs/DFPT_CHUNKS.md`](../../docs/DFPT_CHUNKS.md).

The input and scheduler templates are intentionally incomplete. Replace all
`TODO` fields and provide real preparatory-run XML/pattern data before use.
The tools never submit these scripts or invoke QE themselves.

Adapted from `jiangtong1000/PyEPH` revision
`6c4693acbb69a06a5bc8b0593abde2170ff38843`, under the project's BSD-3-Clause
license. This directory does not depend on that checkout at runtime.
