CDCZip
===========

CDCZip is a deduplication filter meant for use as long range match finder preprocessor
for use alongside other compression algorithms.

Heavily inspired by RZip and the LRZip implementation, but based upon state-of-the-art CDC deduplication approaches
instead of the RZip algorithm.

It technically is an LZ77 compressor, but no other compression like entropy coding or anything of the sort is applied.

Usage example
-------------
- Compress

    `cdczip [inputfile] -dict=[amount_in_mb] > [outputfile]`
  - Example with pipe
  
    `cdczip [inputfile] -dict=[amount_in_mb] | zstd -17 -T0 --no-progress - -o outputfile.zstd`
- Decompress

  `cdczip [inputfile.cdcz] -d=[outputfile]`
  - Example with pipe

    `cdczip [inputfile.cdcz] -d=- | something -`

TODO
-------
- Tons of refactoring, cleaning up, etc.
- Smarter use of multithreading when doing simhashing and delta compression
- Proper file format with header and stuff
- Much more

Contact
-------
You can reach me at nicolas.comerci@fing.edu.uy.

However, please do not contact me by email if another channel would be more appropriate.
In particular, don't ask for features, improvements, bug fixes or format support requests.
The github issues page on the repo is the appropriate channel for those subjects.

Acknowledgements and thanks
-----------

- Álvaro Martín and Guillermo Dufort y Álvarez for supervising my bachelor thesis, which this software is a result of.


- Andrew Tridgell and Con Kolivas for developing the RZip algorithm and it's implementation LRZip respectively; without them CDCZip would not exist.


- Wen Xia, Fan Ni, Binzhaoshuo Wan and their collaborators, their publications were invaluable in my learining on the field of deduplication,
and in particular for their contribution of the FastCDC, SS-CDC and SuperCDC algorithms that CDCZip uses.

Initial FastCDC code ported from https://github.com/iscc/fastcdc-py which was Cython so it was pretty straightforward to do.
Thanks to the ISCC Foundation for providing it with MIT license.

Legal stuff
-----------
- xxHash (https://github.com/Cyan4973/xxHash) by Yann Collet provided under the BSD 2-Clause License

License
-------
Copyright 2024 Nicolás Comerci

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
