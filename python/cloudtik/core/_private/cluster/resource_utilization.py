from abc import abstractmethod
from numbers import Real
from typing import Dict, List, Optional

from cloudtik.core._private.cluster.node_availability_tracker import NodeAvailabilitySummary

NodeResources = Dict[str, Real]
ResourceDemands = List[Dict[str, Real]]


class UtilizationScore:
    """This fancy class just defines the `UtilizationScore` protocol to be
    some type that is a "totally ordered set" (i.e. things that can be sorted).

    What we're really trying to express is

    ```
    UtilizationScore = TypeVar("UtilizationScore", bound=Comparable["UtilizationScore"])
    ```

    but Comparable isn't a real type and, and a bound with a type argument
    can't be enforced (f-bounded polymorphism with contravariance). See Guido's
    comment for more details: https://github.com/python/typing/issues/59.

    This isn't just a `float`. In the case of the default scorer, it's a
    `Tuple[float, float]` which is quite difficult to map to a single number.

    """

    @abstractmethod
    def __eq__(self, other: "UtilizationScore") -> bool:
        pass

    @abstractmethod
    def __lt__(self: "UtilizationScore", other: "UtilizationScore") -> bool:
        pass

    def __gt__(self: "UtilizationScore", other: "UtilizationScore") -> bool:
        return (not self < other) and self != other

    def __le__(self: "UtilizationScore", other: "UtilizationScore") -> bool:
        return self < other or self == other

    def __ge__(self: "UtilizationScore", other: "UtilizationScore") -> bool:
        return not self < other


class UtilizationScorer:
    def __call__(
        node_resources: NodeResources,
        resource_demands: ResourceDemands,
        node_type: str,
        *,
        node_availability_summary: NodeAvailabilitySummary,
    ) -> Optional[UtilizationScore]:
        pass
