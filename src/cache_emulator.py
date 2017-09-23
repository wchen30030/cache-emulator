'''
Cache emulator
Based on Computer Architecture 5th Hennessey, Patterson
AMD Opteron Processor as default
'''

import numpy as np
import math
import random


#global counter for block replacement
counter = 1
#default values to be changed
numDouble = 8
numBlocks = 2
numSets = 512

'''
Memory address in 39 bits; 1 for valid bit.
Take in an address as an int in terms of doubles.
We ignore last three 0 bits to get a multiple of 8 bytes.
Effectively 36 bits for memory address.
What we return will always be integers after converting from base 2.
'''
class Address:

    def __init__(self, address):
        self.addr = address

    def getTag(self):
        return self.addr // (numDouble * numSets)

    def getIndex(self):
        return (self.addr // numDouble) % numSets

    def getOffset(self):
        return self.addr % numDouble

    #get ram's block address by combining offset and index
    def getRamAddr(self):
        return self.addr // numDouble


'''
Stores all memory references in ram.
Initialized before recording instructions.
Called by cache object.
Only need to store 2^22 doubles.
'''
class Ram:

    def __init__(self):
        doubles_in_ram = 2**22
        blocks_in_ram = doubles_in_ram // numDouble

        #initialized ram values
        self.ram = np.arange(doubles_in_ram, dtype=np.float).reshape(blocks_in_ram, numDouble) % 100
        #used for compulsory misses
        self.first_access = np.zeros(blocks_in_ram, dtype=bool)

    #rw='r' for reading, rw='w' for writing
    #only reading affects compulsory miss
    def getBlock(self, address, rw='r'):
        compulsory = False
        addr = address.getRamAddr()
        if rw == 'r':
            if self.first_access[addr] == False:
                self.first_access[addr] = True
                compulsory = True
        return self.ram[addr], compulsory

    #Write to block address with value.
    def setBlock(self, address, block):
        self.ram[address.getRamAddr()] = block

    #Print all data in ram btw given blocks.
    #Each row is one block.
    def printRam(self, start_block, end_block):
        print(self.ram[start_block:end_block+1, :])


'''
Cache memory stored here.
Calls on ram object.
'''
class Cache:

    def __init__(self, replacement):
        self.replace = replacement

        self.cache = np.zeros((numSets, numBlocks, numDouble))
        self.tag = np.zeros((numSets, numBlocks))
        self.valid = np.zeros((numSets, numBlocks), dtype=bool)
        #used for replacement policy based on counter
        #fifo update when created, lru whenever accessed
        self.access = np.zeros((numSets, numBlocks))

        self.cache_ram = Ram()

        self.read_hits = 0
        self.write_hits = 0
        self.write_miss = 0

        self.conflict_miss = 0
        self.compulsory_miss = 0
        self.capacity_miss = 0
        # used for capacity misses
        self.total_blocks = numSets * numBlocks
        self.blocks_filled = 0

    #Called by cpu.
    #Gets double from address.
    #Copies block from ram if not found.
    def getDouble(self, address):
        #check if in cache
        block = self.getBlock(address)
        if block is not None:
            self.read_hits += 1
            return block[address.getOffset()]

        #fetch from ram
        block, compul = self.cache_ram.getBlock(address)

        if compul == True: #compulsory miss
            self.compulsory_miss += 1
        #capacity miss
        elif self.blocks_filled == self.total_blocks:
            self.capacity_miss += 1
        else: #conflict miss
            self.conflict_miss += 1

        #save block to cache
        self.setBlock(address, block)
        return block[address.getOffset()]

    #Called by cpu.
    #If found, sets the double then copies block to ram.
    #If not found, gets block from ram to cache.
    #Then proceed as if found.
    def setDouble(self, address, value):
        #check if in cache
        block = self.getBlock(address)
        if block is not None:
            self.write_hits += 1
            block[address.getOffset()] = value
            self.cache_ram.setBlock(address, block)
            return
        #fetch from ram before writing
        self.write_miss += 1
        block, _ = self.cache_ram.getBlock(address, 'w')
        #save to ram and cache
        block[address.getOffset()] = value
        self.setBlock(address, block)
        return

    #Gets block to rewrite ram data.
    #Called by getDouble and setDouble.
    def getBlock(self, address):
        addr_index = address.getIndex()
        #check if in cache
        for i in range(numBlocks):
            #a match
            if self.valid[addr_index, i] == True and \
                address.getTag() == self.tag[addr_index,i]:
                if self.replace == 'l': #change access time
                    global counter
                    self.access[addr_index,i] = counter
                    counter += 1
                return self.cache[addr_index, i]
        return None

    #Adds new block to cache from ram on miss.
    #May need to evict a block if full.
    def setBlock(self, address, block):
        global counter
        addr_index = address.getIndex()
        #check for open space
        for i in range(numBlocks):
            #found open space
            if self.valid[addr_index,i] == False:
                self.valid[addr_index,i] = True
                self.blocks_filled += 1
                self.access[addr_index,i] = counter
                counter += 1
                self.tag[addr_index,i] = address.getTag()
                self.cache[addr_index,i] = block
                return
        #need to evict
        e = self.evict(address)
        self.access[addr_index,e] = counter
        counter += 1
        self.tag[addr_index,e] = address.getTag()
        self.cache[addr_index,e] = block
        return

    #Returns index of block in set to be replaced.
    def evict(self, address):
        #lru and fifo
        if self.replace == 'l' or self.replace == 'f':
            return np.argmin(self.access[address.getIndex()])
        #random replacement
        return random.randint(0, numBlocks-1)

    #Print all data in cache btw given sets.
    #Each row is one block.
    #Separated by associativity.
    def printCache(self, start_set, end_set):
        print(self.cache[start_set:end_set+1, :, :])


'''
All instructions referenced here.
Calls on cache object.
'''
class Cpu:

    def __init__(self, cache_size_kb, associativity, block_size, replacement):
        #setup cache
        global numBlocks
        global numSets
        cache_size = 1024 * cache_size_kb

        if associativity == 'd':
            numBlocks = 1
            numSets = cache_size // block_size
        elif associativity == 'f':
            numBlocks = cache_size // block_size
            numSets = 1
        else:
            numBlocks = int(associativity)
            numSets = cache_size // (block_size * numBlocks)

        global numDouble
        numDouble = block_size//8

        self.cpu_cache = Cache(replacement)

        #counts instruction 
        self.instr_count = 0

    #Loads from cache.
    #Loads from ram if miss.
    def loadDouble(self, address):
        self.instr_count += 1
        return self.cpu_cache.getDouble(address)
        
    #Stores to cache and ram.
    def storeDouble(self, address, value):
        self.instr_count += 1
        self.cpu_cache.setDouble(address, value)
        
    def addDouble(self, value1, value2):
        self.instr_count += 1
        return value1+value2

    def multDouble(self, value1, value2):
        self.instr_count += 1
        return value1*value2


'''
#sets up cache with different configurations
print('Default: AMD Opteron Processor')
print('Cache size: 64KB')
print('Associativity: 2-way')
print('Block size: 64B (8 doubles)')
print('Replacement strategy: LRU')
#print('Write policy: write-through with write allocate')
print('')

default = input('Change default settings (y/n)? ')
cache_size_kb = 64
associativity = '2'
block_size = 64
replacement = 'l'

if default == 'y':
    #must be positive integer
    cache_size_kb = int(input('New cache size (in KB): '))
    #'d', 'f', or a positive power of 2
    associativity = input('Direct-mapped (d), fully associative (f), or n-way associative (n)? ')
    #a positive power of 2 greater than 8
    block_size = int(input('Block size (in bytes): '))
    replacement = input('Replacement strategy: random (r), LRU (l), or FIFO (f)? ')


cpu = Cpu(cache_size_kb, associativity, block_size, replacement)

# add test cases below or run cache_analyzer.py
'''
