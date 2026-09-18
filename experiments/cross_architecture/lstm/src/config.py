C={}
def configure(c):
 if C and C!=c:raise RuntimeError('CONFIG_ALREADY_BOUND')
 C.update(c)
