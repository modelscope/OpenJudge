# -*- coding: utf-8 -*-
import asyncio


class ConcurrencyManager:
    """
    A manager for controlling concurrency across the entire system.

    This class provides a centralized way to manage and limit the concurrent execution
    of grader evaluations to prevent resource exhaustion.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConcurrencyManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_concurrent: int = 32):
        """Initialize the GraderConcurrencyManager as a singleton."""
        self.set_max_concurrent(max_concurrent)

    def set_max_concurrent(self, max_concurrent: int = 32):
        """
        Set the maximum number of concurrent grader evaluations.

        Args:
            max_concurrent: Maximum number of concurrent evaluations allowed
        """
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")

        self._max_concurrent = max_concurrent
        # Note: We cannot directly change the semaphore's capacity,
        # so we create a new one
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def get_max_concurrent(self) -> int:
        """
        Get the current maximum concurrent limit.

        Returns:
            The maximum number of concurrent evaluations allowed
        """
        return self._max_concurrent

    async def run_with_concurrency_control(self, coro):
        """
        Run a coroutine with concurrency control.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        async with self._semaphore:
            return await coro

    @property
    def current_semaphore(self):
        """
        Get the current semaphore being used for concurrency control.

        Returns:
            The asyncio.Semaphore instance
        """
        return self._semaphore
