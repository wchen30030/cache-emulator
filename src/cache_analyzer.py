
'''
Cache analyzer
requires cache_emulator.py
tests cache emulator against optimization metrics:
cache size, associativity, block size, replacement policy
from Computer Architecture 5th by Hennessey, Patterson
on dot product and matrix multiplication and its optimizations
'''

#exec(open("cache_emulator.py").read())

import cache_emulator as ce


#dot product analysis
def dot_product(cache, assoc, block, replace):
    cpu = ce.Cpu(cache, assoc, block, replace)
    n = 10000
    r = 0
    for i in range(n):
        x = cpu.loadDouble(ce.Address(i))
        y = cpu.loadDouble(ce.Address(n+i))
        z = cpu.multDouble(x, y)
        r = cpu.addDouble(r, z)
    cpu.storeDouble(ce.Address(2*n), r)
    print('dot product',cache,assoc,block,replace)
    print(cpu.instr_count, cpu.cpu_cache.read_hits, cpu.cpu_cache.compulsory_miss, cpu.cpu_cache.capacity_miss, cpu.cpu_cache.conflict_miss, cpu.cpu_cache.write_hits, cpu.cpu_cache.write_miss)
    return cpu.instr_count, cpu.cpu_cache.read_hits, cpu.cpu_cache.compulsory_miss,\
            cpu.cpu_cache.capacity_miss, cpu.cpu_cache.conflict_miss, \
            cpu.cpu_cache.write_hits, cpu.cpu_cache.write_miss


#standard matrix multiplication
def mxm(cache, assoc, block, replace):
    cpu = ce.Cpu(cache, assoc, block, replace)
    n = 100
    for i in range(n):
        for j in range(n):
            r = 0
            for k in range(n):
                x = cpu.loadDouble(ce.Address(i*n+k))
                y = cpu.loadDouble(ce.Address(n**2+j+k*n))
                z = cpu.multDouble(x, y)
                r = cpu.addDouble(r, z)
            cpu.storeDouble(ce.Address(2*n**2+i*n+j), r)
    print('mxm',cache,assoc,block,replace)
    print(cpu.instr_count, cpu.cpu_cache.read_hits, cpu.cpu_cache.compulsory_miss, cpu.cpu_cache.capacity_miss, cpu.cpu_cache.conflict_miss, cpu.cpu_cache.write_hits, cpu.cpu_cache.write_miss)
    return cpu.instr_count, cpu.cpu_cache.read_hits, cpu.cpu_cache.compulsory_miss,\
            cpu.cpu_cache.capacity_miss, cpu.cpu_cache.conflict_miss, \
            cpu.cpu_cache.write_hits, cpu.cpu_cache.write_miss
 

#matrix multiplication after loop interchange
def mxm_loop(cache, assoc, block, replace):
    cpu = ce.Cpu(cache, assoc, block, replace)
    n = 100
    for j in range(n):
        for i in range(n):
            r = 0
            for k in range(n):
                x = cpu.loadDouble(ce.Address(i*n+k))
                y = cpu.loadDouble(ce.Address(n**2+j+k*n))
                z = cpu.multDouble(x, y)
                r = cpu.addDouble(r, z)
            cpu.storeDouble(ce.Address(2*n**2+i*n+j), r)
    print('mxm_loop',cache,assoc,block,replace)
    print(cpu.instr_count, cpu.cpu_cache.read_hits, cpu.cpu_cache.compulsory_miss, cpu.cpu_cache.capacity_miss, cpu.cpu_cache.conflict_miss, cpu.cpu_cache.write_hits, cpu.cpu_cache.write_miss)
    return cpu.instr_count, cpu.cpu_cache.read_hits, cpu.cpu_cache.compulsory_miss,\
            cpu.cpu_cache.capacity_miss, cpu.cpu_cache.conflict_miss, \
            cpu.cpu_cache.write_hits, cpu.cpu_cache.write_miss


#matrix multiplication with blocking
def mxm_blocking(cache, assoc, block, replace):
    cpu = ce.Cpu(cache, assoc, block, replace)
    n = 100
    B = 10 #blocking size
    #setup
    for i in range(2*n**2,3*n**2):
        cpu.storeDouble(ce.Address(i),0)
    #reset to not mess with stats
    cpu.instr_count = 0
    cpu.cpu_cache.read_hits = 0
    cpu.cpu_cache.compulsory_miss = 0
    cpu.cpu_cache.capacity_miss = 0
    cpu.cpu_cache.conflict_miss = 0
    cpu.cpu_cache.write_hits = 0
    cpu.cpu_cache.write_miss = 0
    cpu.cpu_cache.blocks_filled = 0
    cpu.cpu_cache.valid *= False
    cpu.cpu_cache.cache_ram.first_access *= False
    cpu.cpu_cache.cache *= 0
    cpu.cpu_cache.tag *= 0 
    cpu.cpu_cache.access *= 0

    for jj in range(0,n,B):
        for kk in range(0,n,B):
            for i in range(n):
                for j in range(jj, min(jj+B, n)):
                    r = 0
                    for k in range(kk, min(kk+B, n)):
                        x = cpu.loadDouble(ce.Address(i*n+k))
                        y = cpu.loadDouble(ce.Address(n**2+j+k*n))
                        z = cpu.multDouble(x, y)
                        r = cpu.addDouble(r, z)
                    xx = cpu.loadDouble(ce.Address(2*n**2+i*n+j))
                    xx = cpu.addDouble(xx, r)
                    cpu.storeDouble(ce.Address(2*n**2+i*n+j),xx)
    print('mxm_blocking',cache,assoc,block,replace)
    print(cpu.instr_count, cpu.cpu_cache.read_hits, cpu.cpu_cache.compulsory_miss, cpu.cpu_cache.capacity_miss, cpu.cpu_cache.conflict_miss, cpu.cpu_cache.write_hits, cpu.cpu_cache.write_miss)
    return cpu.instr_count, cpu.cpu_cache.read_hits, cpu.cpu_cache.compulsory_miss,\
            cpu.cpu_cache.capacity_miss, cpu.cpu_cache.conflict_miss, \
            cpu.cpu_cache.write_hits, cpu.cpu_cache.write_miss



print('Test cache associativity')
print('Algorithm    Cache size    Associativity    Block size    Replacement policy')
print('instr_count read_hit compulsory capacity conflict write_hit write_miss')
cache_assoc_dot = [dot_product(64,'d',64,'l'), dot_product(64,'2',64,'l'), dot_product(64,'4',64,'l'), dot_product(64,'8',64,'l'), dot_product(64,'16',64,'l'), dot_product(64,'f',64,'l')]

print('')

cache_assoc_mxm = [mxm(64,'d',64,'l'), mxm(64,'2',64,'l'), mxm(64,'4',64,'l'), mxm(64,'8',64,'l')]
#, mxm(64,'16',64,'l'), mxm(64,'f',64,'l')]
#same output as 8-way assoc but longer to run


print('Test replacement policy')
print('Algorithm    Cache size    Associativity    Block size    Replacement policy')
print('instr_count read_hit compulsory capacity conflict write_hit write_miss')
replacement_dot = [dot_product(64,'2',64,'l'), dot_product(64,'2',64,'f'),  dot_product(64,'2',64,'r')]

print('')

replacement_mxm = [mxm(64,'2',64,'l'), mxm(64,'2',64,'f'), mxm(64,'2',64,'r')]


print('Test block size')
print('Algorithm    Cache size    Associativity    Block size    Replacement policy')
print('instr_count read_hit compulsory capacity conflict write_hit write_miss')
block_size_dot = [dot_product(64,'2',16,'l'), dot_product(64,'2',32,'l'), dot_product(64,'2',64,'l'), dot_product(64,'2',128,'l')]

print('')

block_size_mxm = [mxm(64,'2',16,'l'), mxm(64,'2',32,'l'), mxm(64,'2',64,'l'), mxm(64,'2',128,'l')]


print('Test cache size')
print('Algorithm    Cache size    Associativity    Block size    Replacement policy')
print('instr_count read_hit compulsory capacity conflict write_hit write_miss')
cache_size_dot = [dot_product(4,'2',64,'l'), dot_product(8,'2',64,'l'), dot_product(16,'2',64,'l'), dot_product(32,'2',64,'l'), dot_product(64,'2',64,'l'), dot_product(128,'2',64,'l')]

print('')

cache_size_mxm = [mxm(4,'2',64,'l'), mxm(8,'2',64,'l'), mxm(16,'2',64,'l'), mxm(32,'2',64,'l'), mxm(64,'2',64,'l'), mxm(128,'2',64,'l')]


print('Test compiler optimization for matrix multiplication')
print('Algorithm    Cache size    Associativity    Block size    Replacement policy')
print('instr_count read_hit compulsory capacity conflict write_hit write_miss')
mxm = [mxm(64,'2',64,'l'), mxm_loop(64,'2',64,'l'), mxm_blocking(64,'2',64,'l')]

