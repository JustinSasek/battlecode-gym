from gym.spaces import Space
from typing import List as TypingList
from collections.abc import MutableSequence
import numpy as np


class List(Space[TypingList[Space]], MutableSequence):
    """
    A list of simpler spaces.

    Example usage:
    self.observation_space = List([spaces.Discrete(2), spaces.Discrete(3)])
    """

    def __init__(
            self,
            spaces: list[Space] | None = None,
            seed: list | int | None = None,
    ) -> None:
        assert isinstance(spaces, list), 'spaces must be a list'

        self.spaces = spaces if spaces is not None else []
        for space in spaces:
            assert isinstance(
                space, Space
            ), "Values of the list should be instances of gym.Space"
        super().__init__(
            None, None, seed  # type: ignore
        )  # None for shape and dtype, since it'll require special handling


    def seed(self, seed: list | int | None = None) -> list:
        seeds = []
        if isinstance(seed, list):
            assert len(seed) == len(self.spaces), print(
                "Seed list is different size from spaces list.",
            )
            for space, seed in zip(self.spaces, seed):
                seeds += space.seed(seed)
        elif isinstance(seed, int):
            seeds = super().seed(seed)
            try:
                subseeds = self.np_random.choice(
                    np.iinfo(int).max,
                    size=len(self.spaces),
                    replace=False,  # unique subseed for each subspace
                )
            except ValueError:
                subseeds = self.np_random.choice(
                    np.iinfo(int).max,
                    size=len(self.spaces),
                    replace=True,  # we get more than INT_MAX subspaces
                )

            for subspace, subseed in zip(self.spaces, subseeds):
                seeds.append(subspace.seed(int(subseed))[0])
        elif seed is None:
            for space in self.spaces:
                seeds += space.seed(seed)
        else:
            raise TypeError("Passed seed not of an expected type: list or int or None")

        return seeds

    def sample(self) -> list:
        return [space.sample() for space in self.spaces]

    def contains(self, x) -> bool:
        if not isinstance(x, list) or len(x) != len(self.spaces):
            return False
        for i, space in enumerate(self.spaces):
            if not space.contains(x[i]):
                return False
        return True

    def __getitem__(self, i):
        return self.spaces[i]

    def __setitem__(self, i, item):
        self.spaces[i] = item

    def __delitem__(self, i):
        del self.spaces[i]

    def __iter__(self):
        yield from self.spaces

    def __len__(self) -> int:
        return len(self.spaces)

    def __repr__(self) -> str:
        return (
                "List("
                + ", ".join([str(space) for space in self.spaces])
                + ")"
        )

    def insert(self, i: int, space: Space) -> None:
        self.spaces.insert(i, space)

    def append(self, space: Space) -> None:
        self.spaces.append(space)

    def to_jsonable(self, sample_n: list) -> list:
        # serialize as list-repr of vectors
        return [
            space.to_jsonable([sample[i] for sample in sample_n])
            for i, space in enumerate(self.spaces)
        ]

    def from_jsonable(self, sample_n: list[list]) -> list:
        list_of_list: list[list] = []  # first axis is spaces, second is samples
        for i, space in enumerate(self.spaces):
            list_of_list[i] = space.from_jsonable(sample_n[i])
        ret = []
        n_elements = len(next(iter(list_of_list)))
        for i in range(n_elements):
            entry = []
            for space_samples in list_of_list:
                entry.append(space_samples[i])
            ret.append(entry)
        return ret
