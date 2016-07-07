#!/usr/bin/env python
"""Pseudo infinite length array with integrated garbage collection"""


import numpy as np
import math
import collections


class RingNdarray:
    def __init__(self, block_shape, block_init_fct=np.zeros,
                 init_kwargs={},
                 block_count=2): 
        """)
        block_shape: shape of each block as a tuple
        """
        self.block_shape = block_shape
        self.blen = block_shape[0]
        self.block_count = block_count
        self.block_init_fct = block_init_fct
        self.init_kwargs = init_kwargs

        self.arr = block_init_fct((self.blen*block_count,)+ block_shape[1:], **init_kwargs)

        self.bounds = np.array([x*self.blen for x in range(block_count+1)])
        self.truebounds = collections.deque([x*self.blen for x in range(block_count)])

    def convert_slice(self, s):
        """Converts a slice into an index array that can be used directly in self.arr"""
        bounds = self.bounds
        if s.start < bounds[0] or s.stop > bounds[-1]:
            raise Exception('Slice out of bounds')

        pairlist = []
        for k in range(len(bounds)-1):
            if s.start >= bounds[k+1] or s.stop <= bounds[k]:
                continue
            start = max(bounds[k], s.start)
            stop = min(bounds[k+1], s.stop)

            truestart = start%self.blen + self.truebounds[k]
            truestop = stop%self.blen
            if truestop == 0:
                truestop = self.blen
            truestop += self.truebounds[k]

            pairlist.append(tuple([truestart, truestop]))

        return np.concatenate([np.arange(*pair) for pair in pairlist])

    def checkidx(self, idx):
        """Raises exception if idx is below earliest block or after next max.
        Raises true/false if index is above current max"""
        if idx < self.bounds[0]:
            raise ValueError("Index below the current minimum value")
        if idx >= self.bounds[-1]+self.blen:
            raise ValueError("Index too high, even after rotating the blocks once")
        return idx <= self.bounds[-1]

    def checkslice(self, s):
        """Checks if slice is in range"""
        return self.checkidx(s.start) and self.checkidx(s.stop)

    def get_mask(self, input_key):
        key = input_key[0]
        remainder_key = input_key[1:]
        if isinstance(key, range):
            key = slice(key.start, key.stop, key.step)

        if isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise Exception('Steps other than 1 are unsupported')

            if not self.checkslice(key):
                self.rotate()
                self.checkslice(key) # Trigger exception handling

            mask = self.convert_slice(key)
            return (mask,)+remainder_key
        else:
            raise Exception('Non-slice getitem not implemented')

    def rotate(self):
        """Rotates the bounds of the array and initialized the new block"""
        self.truebounds.rotate(-1)
        self.bounds += self.blen
        self.arr[self.truebounds[-1]:self.truebounds[-1]+self.blen, ...] = self.block_init_fct(
            self.block_shape, **self.init_kwargs)

    def __getitem__(self, input_key):
        return self.arr[self.get_mask(input_key)]

    def __setitem__(self, input_key, value):
        mask = self.get_mask(input_key)
        self.arr[mask] = value




if __name__ == '__main__':

    chan = RingNdarray((5,2), block_count=3)
    chan[9:16,0] = np.ones(7)

    print(chan.arr)
    print(chan[5:20,:])


