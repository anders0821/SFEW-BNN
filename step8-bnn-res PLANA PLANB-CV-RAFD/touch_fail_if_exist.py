import sys

if __name__ == "__main__":
    assert(len(sys.argv)==2)
    fn = sys.argv[1]
    try:
        f = open(fn, 'wx') # fail if exist, for allowing parallel jobs on single machine
	f.close()
    except:
        exit(1)

