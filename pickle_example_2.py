import pickle
def say_hello(msg):
    print("RUNNING:", msg)
class Demo:
    def __reduce__(self):
        return (say_hello, ("hi",))
open("demo.pkl", "wb").write(pickle.dumps(Demo()))
