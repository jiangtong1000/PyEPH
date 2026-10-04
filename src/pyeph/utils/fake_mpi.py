"""Small serial communicator for historical interfaces; no MPI dependency."""

import numpy as np


class FakeReq:
    def wait(self):
        return None


class FakeComm:
    rank = 0
    size = 1

    def __init__(self):
        self._messages = {}

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def barrier(self):
        return None

    Barrier = barrier

    def bcast(self, value, root=0):
        return value

    Bcast = bcast

    def gather(self, value, root=0):
        return [value]

    allgather = gather

    def allreduce(self, value, op=None, root=0):
        return np.copy(value) if isinstance(value, np.ndarray) else value

    def Gather(self, sendbuf, recvbuf, root=0):
        recvbuf[...] = sendbuf

    Allgather = Gather

    def Allreduce(self, sendbuf, recvbuf, op=None, root=0):
        recvbuf[...] = sendbuf

    Reduce = Allreduce
    Scatter = Gather
    Scatterv = Gather

    def scatter(self, value, root=0):
        if len(value) != 1:
            raise ValueError("serial scatter requires one rank's payload")
        return value[0]

    def Split(self, color=0, key=0):
        return self

    def isend(self, value, dest=0, tag=0):
        if dest not in (None, 0):
            raise ValueError("a serial communicator has only rank zero")
        self._messages[tag] = np.copy(value) if isinstance(value, np.ndarray) else value
        return FakeReq()

    Isend = isend

    def recv(self, source=0, root=0, tag=0):
        if source not in (None, 0):
            raise ValueError("a serial communicator has only rank zero")
        if tag not in self._messages:
            raise RuntimeError("serial receive has no corresponding send")
        return self._messages.pop(tag)

    def Recv(self, recvbuff, source=0, root=0, tag=0):
        recvbuff[...] = self.recv(source=source, root=root, tag=tag)


class MPI:
    COMM_WORLD = FakeComm()
    SUM = None
    COMM_SPLIT_TYPE_SHARED = None
    COMM_TYPE_SHARED = None
    DOUBLE = None
    INT64_T = None
    Win = None
    IntraComm = FakeComm
