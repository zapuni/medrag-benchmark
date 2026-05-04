from abc import ABC, abstractmethod


class HealthCheckable(ABC):
    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the adapter is healthy."""
        raise NotImplementedError
