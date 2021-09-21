from sys import argv
from os.path import basename

if len(argv) != 2:
    print 'usage: python2 %s cmudict_file' % argv[0]
    exit()

## Get everything from the CMU dict file

wl  = [ '<eps>', '<s>', '</s>', '<unk>' ]
pl  = set(wl)
cmu = { w: [ w ] for w in wl }

for line in open(argv[1], 'r'):
    if line.startswith(';'): 
        continue
    c = line.strip().split()
    wl.append(c[0])
    pl.update(c[1:])
    cmu[c[0]] = c[1:]

cmufilename = basename(argv[1])

## Dump the wordlist

with open('wl.%s' % cmufilename, 'w') as fwl:
    fwl.write('\n'.join([ '%s\t%d' % (w,i) for i,w in enumerate(wl) ]) + '\n')

## Dump phonelist

with open('pl.%s' % cmufilename, 'w') as fpl:
    fpl.write('\n'.join([ '%s\t%d' % (p,i) for i,p in enumerate(sorted(pl)) ]) + '\n')

## Build and dump FSM

with open('%s.fsm.txt' % cmufilename, 'w') as ffsm:
    statecount = 1
    # Skip <eps>
    for w in wl[1:]:
        ffsm.write( "0\t%d\t%s\t%s\n" % (statecount, w, cmu[w][0]) )
        for p in cmu[w][1:]:
            ffsm.write( "%d\t%d\t<eps>\t%s\n" % (statecount, statecount+1, p) )
            statecount += 1
        ffsm.write( "%d\t0\t<eps>\t<eps>\n" % statecount )
        statecount +=1
    ## THIS LINE IS CRITICAL! SERIOUSLY!
    ffsm.write('0\n')
