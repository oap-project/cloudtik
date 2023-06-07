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

        distributor.validate_same_slots()

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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
