from __future__ import annotations
from abc import ABC, abstractmethod

from lob_sim.core.models import Action, MarketObservation


class BaseArchetype(ABC):
    """
    Abstract base class for all agent archetypes.

    Every archetype must implement `act`, which receives the current market
    observation and returns a (possibly empty) list of actions to submit.
    `reset` is called between simulation scenarios to clear internal state.
    """

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @abstractmethod
    def act(self, obs: MarketObservation) -> list[Action]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
