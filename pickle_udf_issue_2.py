import pickle, pickletools

def say_hello(msg):
    print("RUNNING during load:", msg)
    return msg

class Demo:
    def __reduce__(self):               # tells pickle how to rebuild this object
        return (say_hello, ("hi from pickle",))

data = pickle.dumps(Demo())
print("dumped, nothing has run yet")
pickletools.dis(data)                    # shows say_hello named as a string
obj = pickle.loads(data)
print("loaded:", obj)                    # obj is the string, not a Demo
