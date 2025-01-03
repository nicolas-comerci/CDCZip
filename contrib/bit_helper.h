/* Copyright 2018 Dirk Steinke

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef BIT_HELPER_H
#define BIT_HELPER_H

#include <cstdint>

unsigned bitLength(unsigned value);
uint64_t bitReverse(const uint64_t value, const unsigned bits);
unsigned bitLeadingZeroes(const unsigned value);
unsigned bitTrailingZeroes(const unsigned value);

#endif /* BIT_HELPER_H */
