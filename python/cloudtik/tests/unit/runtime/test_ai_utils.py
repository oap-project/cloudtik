import unittest

import pytest

from cloudtik.runtime.ai.runner.util.distributor import Distributor


class TestAIUtils(unittest.TestCase):
    def test_distributor(self):
        distributor = Distributor(
            nnodes=2,
            nproc_per_node=2,
            hosts="10.0.0.1, 10.0.0.2"
        )

        assert distributor.num_proc == 4
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 2
        assert distributor.hosts[0]["ip"] == "10.0.0.1"
        assert distributor.hosts[0]["slots"] == 2
        assert distributor.hosts[1]["ip"] == "10.0.0.2"
        assert distributor.hosts[1]["slots"] == 2

        distributor = Distributor(
            num_proc=4,
            hosts="10.0.0.1, 10.0.0.2"
        )
        assert distributor.num_proc == 4
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 2
        assert distributor.hosts[0]["ip"] == "10.0.0.1"
        assert distributor.hosts[0]["slots"] == 2
        assert distributor.hosts[1]["ip"] == "10.0.0.2"
        assert distributor.hosts[1]["slots"] == 2

        distributor = Distributor(
            hosts="10.0.0.1:2, 10.0.0.2:2"
        )

        assert distributor.num_proc == 4
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 2
        assert distributor.hosts[0]["ip"] == "10.0.0.1"
        assert distributor.hosts[0]["slots"] == 2
        assert distributor.hosts[1]["ip"] == "10.0.0.2"
        assert distributor.hosts[1]["slots"] == 2

        distributor = Distributor(
            hosts="10.0.0.1:2, 10.0.0.2:4"
        )

        assert distributor.num_proc == 6
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 3
        assert distributor.hosts[0]["ip"] == "10.0.0.1"
        assert distributor.hosts[0]["slots"] == 2
        assert distributor.hosts[1]["ip"] == "10.0.0.2"
        assert distributor.hosts[1]["slots"] == 4

        distributor = Distributor(
            num_proc=6,
            nproc_per_node=2,
            hosts="10.0.0.1:6, 10.0.0.2"
        )
        assert distributor.num_proc == 6
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 3
        assert distributor.hosts[0]["ip"] == "10.0.0.1"
        assert distributor.hosts[0]["slots"] == 6
        assert distributor.hosts[1]["ip"] == "10.0.0.2"
        assert distributor.hosts[1]["slots"] == 2

        distributor = Distributor(
            nproc_per_node=2,
            hosts="10.0.0.1, 10.0.0.2"
        )
        assert distributor.num_proc == 4
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 2
        assert distributor.hosts[0]["ip"] == "10.0.0.1"
        assert distributor.hosts[0]["slots"] == 2
        assert distributor.hosts[1]["ip"] == "10.0.0.2"
        assert distributor.hosts[1]["slots"] == 2

        distributor = Distributor(
            num_proc=4,
            hosts="10.0.0.1:3, 10.0.0.2:3"
        )
        assert distributor.num_proc == 4
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 2
        assert distributor.hosts[0]["ip"] == "10.0.0.1"
        assert distributor.hosts[0]["slots"] == 3
        assert distributor.hosts[1]["ip"] == "10.0.0.2"
        assert distributor.hosts[1]["slots"] == 3

        # for this case, we default nproc_per_node == sockets_per_node
        distributor = Distributor(
            hosts="10.0.0.1, 10.0.0.2"
        )
        assert not distributor.resolved
        assert distributor.nnodes == 2

        distributor.resolve(3)
        assert distributor.num_proc == 6
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 3
        assert distributor.hosts[0]["ip"] == "10.0.0.1"
        assert distributor.hosts[0]["slots"] == 3
        assert distributor.hosts[1]["ip"] == "10.0.0.2"
        assert distributor.hosts[1]["slots"] == 3

        distributor.check_same_slots()

        distributor.resolve(4)
        assert distributor.num_proc == 6
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 3
        assert distributor.hosts[0]["slots"] == 3
        assert distributor.hosts[1]["slots"] == 3

        distributor.resolve(4, force=True)
        assert distributor.num_proc == 8
        assert distributor.nnodes == 2
        assert distributor.nproc_per_node == 4
        assert distributor.hosts[0]["slots"] == 4
        assert distributor.hosts[1]["slots"] == 4

        try:
            distributor = Distributor(
                num_proc=6,
                nproc_per_node=2,
                hosts="10.0.0.1, 10.0.0.2"
            )
            self.fail("Should raise error: not enough slots.")
        except ValueError as e:
            pass

        # local node case
        distributor = Distributor()
        assert distributor.num_proc is None
        assert distributor.nnodes is None
        assert distributor.nproc_per_node is None
        assert distributor.hosts is None

        distributor.resolve(nproc_per_node=2, nnodes=1, check=True)
        assert distributor.num_proc == 2
        assert distributor.nnodes == 1
        assert distributor.nproc_per_node == 2

        # resource scheduler case: nothing
        distributor = Distributor()
        distributor.resolve(nproc_per_node=1)
        assert distributor.num_proc is None
        assert distributor.nnodes is None
        assert distributor.nproc_per_node == 1

        # resource scheduler case: num_proc
        distributor = Distributor(
            num_proc=6
        )
        assert distributor.num_proc == 6
        assert distributor.nnodes is None
        assert distributor.nproc_per_node is None

        distributor.resolve(nproc_per_node=2)
        assert distributor.num_proc == 6
        assert distributor.nnodes == 3
        assert distributor.nproc_per_node == 2
        distributor.check_resolved()

        # resource scheduler case: nnodes
        distributor = Distributor(
            nnodes=3
        )
        assert distributor.num_proc is None
        assert distributor.nnodes == 3
        assert distributor.nproc_per_node is None

        distributor.resolve(nproc_per_node=2)
        assert distributor.num_proc == 6
        assert distributor.nnodes == 3
        assert distributor.nproc_per_node == 2
        distributor.check_resolved()

        # resource scheduler case: everything case 1
        distributor = Distributor(
            num_proc=6,
            nnodes=3
        )
        assert distributor.num_proc == 6
        assert distributor.nnodes == 3
        assert distributor.nproc_per_node == 2
        distributor.check_resolved()

        distributor.resolve(nproc_per_node=3)
        assert distributor.num_proc == 6
        assert distributor.nnodes == 3
        assert distributor.nproc_per_node == 2
        distributor.check_resolved()

        # resource scheduler case: everything case 2
        distributor = Distributor(
            nnodes=3,
            nproc_per_node=2
        )
        assert distributor.num_proc == 6
        assert distributor.nnodes == 3
        assert distributor.nproc_per_node == 2
        distributor.check_resolved()

        distributor.resolve(nproc_per_node=3)
        assert distributor.num_proc == 6
        assert distributor.nnodes == 3
        assert distributor.nproc_per_node == 2
        distributor.check_resolved()

        # conflict parameters for hosts based distributed
        distributor = Distributor(
            nnodes=3,
            nproc_per_node=2
        )
        try:
            distributor.check_distributed_with_hosts()
            self.fail("Should raise error: distributed without hosts.")
        except ValueError as e:
            pass


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
