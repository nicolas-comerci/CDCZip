from functools import reduce
from collections import defaultdict
import random

def hamming_syndrome(bits):
    return reduce(
        # Reduce by XOR
        lambda x, y: x ^ y,
        # All indices of active bits
        [i for (i, b) in enumerate(bits) if b]
    )

def hamming_syndrome_bytes(bits: bytes):
    result = 0
    mask = 128
    for i in range(64):
        bit = mask & bits
        if bit != 0:
            result ^= i
        mask >>= 1
        
    return result


def hamming_syndrome_bytes_v2(bits: bytes):
    result = 0
    mask = 1
    for i in range(64):
        bit = mask & bits
        if bit != 0:
            result ^= i
        mask <<= 1
        
    return result

def generate_hamming_directions(bitlength):
    directions = set()
    # range(1, bitlength - 1) as there is no way to pick 2bits when on the last bit
    # and the first bit doesnt participate on hamming codes
    for i in range(1, bitlength - 1):
        for x in range(i + 1, bitlength):
            new_direction = [0] * bitlength
            new_direction[i] = 1
            new_direction[x] = 1
            syndrome = hamming_syndrome(new_direction)
            new_direction[syndrome] = 1
            directions.add(tuple(new_direction))
    return directions

def index_hamming_directions(directions):
    index = {}
    for direction in directions:
        bits = []
        for i, bit in enumerate(direction):
            if bit == 1:
                bits.append(i)

        if bits[0] not in index:
            index[bits[0]] = {}
        index[bits[0]][bits[1]] = bits[2]
        # We could also find situations where the first bit of the direction
        # is 0, but the other two are not so using the direction is needed
        if bits[1] not in index:
            index[bits[1]] = {}
        index[bits[1]][bits[2]] = bits[0]
        # Its also possible for the middle bit to be 0, but the first and last
        # of the direction being 1 and thus the direction being needed
        index[bits[0]][bits[2]] = bits[1]
    return index

def decompose_vector(index, orig_vector):
    if hamming_syndrome(orig_vector) != 0:
        raise Exception("Vector must be a hamming codeword")
    directions_used = defaultdict(int)
    vector_len = len(orig_vector)
    zero_vec = [0] * vector_len
    vector = [*orig_vector]
    while vector != zero_vec:
        weight_3_dir = None
        weight_2_dir = None
        for i, bit in enumerate(vector):
            if i not in index:
                continue
            subindex = index[i]
            for second_bit_i, third_bit_i in subindex.items():
                dir_weight = vector[i] + vector[second_bit_i] + vector[third_bit_i]
                if dir_weight == 3:
                    weight_3_dir = (i, second_bit_i, third_bit_i)
                    break
                if dir_weight == 2 and weight_2_dir is None:
                    weight_2_dir = (i, second_bit_i, third_bit_i)
            if weight_3_dir is not None:
                break

        if not weight_3_dir and not weight_2_dir:
            raise Exception("Wtf no direction found when decomposing vector")
        if weight_3_dir:
            (first_bit_i, second_bit_i, third_bit_i) = weight_3_dir
        else:
            (first_bit_i, second_bit_i, third_bit_i) = weight_2_dir
        vector[first_bit_i] ^= 1
        vector[second_bit_i] ^= 1
        vector[third_bit_i] ^= 1
        direction = [0] * vector_len
        direction[first_bit_i] = 1
        direction[second_bit_i] = 1
        direction[third_bit_i] = 1
        directions_used[tuple(direction)] += 1
        
        #first_bit = None
        #for i, bit in enumerate(vector):
        #    if bit == 1:
        #        first_bit = i
        #        break
        #vector_tail = vector[(first_bit + 1):]
        #for i, bit in enumerate(vector_tail):
        #    adjusted_i = i + first_bit + 1
        #    if first_bit in index and adjusted_i in index[first_bit]:
        #        if vector[first_bit] + vector[adjusted_i] + vector[index[first_bit][adjusted_i]] >= 2:
        #            vector[first_bit] ^= 1
        #            vector[adjusted_i] ^= 1
        #            vector[index[first_bit][adjusted_i]] ^= 1
        #            direction = [0] * vector_len
        #            direction[first_bit] = 1
        #            direction[adjusted_i] = 1
        #            direction[index[first_bit][adjusted_i]] = 1
        #            directions_used[tuple(direction)] += 1
        #            break
    return directions_used

def test_64bit_vector_decomposition():
    directions = generate_hamming_directions(64)
    index = index_hamming_directions(directions)
    for _ in range(10000):
        rand_vec = []
        for i in range(64):
            rand_vec.append(random.randint(0,1))
        rand_vec[0] = 0  # First bit doesnt participate in hamming codes
        syndrome = hamming_syndrome(rand_vec)
        if syndrome != 0:
            rand_vec[syndrome] ^= 1
        if hamming_syndrome(rand_vec) != 0:
            raise Exception("Somehow random vector couldnt get corrected to codeword")
        vector_directions = decompose_vector(index, rand_vec)
        print(f"Successfully decomposed vector! Found {len(vector_directions)} directions")

if __name__ == "__main__":
    directions = generate_hamming_directions(64)
    index = index_hamming_directions(directions)
    #rand_vec = []
    #for i in range(64):
    #    rand_vec.append(random.randint(0,1))
    #syndrome = hamming_syndrome(rand_vec)
    #if syndrome != 0:
    #    rand_vec[syndrome] ^= 1
    
    rand_vec = [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
    vector_directions = decompose_vector(index, rand_vec)
    vector_directions_set = set(vector_directions.keys())

    rand_vec[8] = 0
    rand_vec[49] = 0
    syndrome = hamming_syndrome(rand_vec)
    rand_vec[syndrome] ^= 1
    vector_directions2 = decompose_vector(index, rand_vec)
    vector_directions2_set = set(vector_directions2.keys())
    common_directions = vector_directions_set & vector_directions2_set
    len(common_directions)
