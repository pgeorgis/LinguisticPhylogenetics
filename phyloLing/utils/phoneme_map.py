from typing import NewType, Tuple

type Phoneme = NewType('Phoneme', str)

type MultiPhoneme = Phoneme | tuple[Phoneme, ...]


class DoubleMap[TKey, TValue]:
    def __init__(self, default_value: TValue):
        self.default_value = default_value
        self.values: dict[TKey, dict[TKey, TValue]] = {}

    def get_value(self, primary_key: TKey,
                  secondary_key: TKey) -> TValue:
        first_map = self.values.get(primary_key)
        if first_map is None:
            self.set_value(primary_key, secondary_key, self.default_value)
            return self.default_value
        second_map_value = first_map.get(secondary_key)
        if second_map_value is None:
            self.set_value(primary_key, secondary_key, self.default_value)
            return self.default_value
        return second_map_value

    def get_value_or_default(self, primary_key: TKey,
                            secondary_key: TKey,
                            default) -> TValue:
        first_map = self.values.get(primary_key)
        if first_map is None:
            return default
        return first_map.get(secondary_key, default)

    def set_value(self, primary_key: TKey,
                  secondary_key: TKey,
                  value: TValue):
        if primary_key not in self.values:
            self.values[primary_key] = {}
        self.values[primary_key][secondary_key] = value

    def is_empty(self):
        return len(self.values) == 0

    def has_value(self, primary_key: TKey,
                  secondary_key: TKey) -> bool:
        return primary_key in self.values and secondary_key in self.values[primary_key]

    def get_key_pairs(self) -> list[Tuple[MultiPhoneme, MultiPhoneme]]:
        for primary_key in self.values:
            for secondary_key in self.values[primary_key]:
                yield primary_key, secondary_key

    def get_primary_keys(self) -> list[TKey]:
        return list(self.values.keys())

    def get_secondary_keys(self, primary_key: TKey) -> list[TKey]:
        if primary_key not in self.values:
            return []
        return list(self.values[primary_key].keys())

    def length(self):
        return len(self.values)

    def __len__(self):
        return self.length()

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return str(self.values)

    def __eq__(self, other):
        if isinstance(other, DoubleMap):
            return self.values == other.values
        if isinstance(other, dict):
            return self.values == other
        return False

class IntegerPhonemeMap:
    def __init__(self, default_score: int = 0):
        self.internal_map = DoubleMap[MultiPhoneme, int](default_score)

    def get_value(self, phoneme: MultiPhoneme, other_phoneme: MultiPhoneme) -> int:
        return self.internal_map.get_value(phoneme, other_phoneme)

    def increment_by_one(self, phoneme: Phoneme, other_phoneme: Phoneme) -> None:
        current_score = self.get_value(phoneme, other_phoneme)
        self.internal_map.set_value(phoneme, other_phoneme, current_score + 1)

    def get_key_pairs(self):
        return self.internal_map.get_key_pairs()

type SomeNumber = int | float

class PhonemeMap:
    def __init__(self, default_value: SomeNumber = 0):
        self.default_value: SomeNumber = default_value
        self.internal_map = DoubleMap[MultiPhoneme, SomeNumber](default_value)

    def __getitem__(self, **args) -> SomeNumber:
        raise NotImplementedError("Use get_value instead")

    def __setitem__(self, **args):
        raise NotImplementedError("Use set_value instead")

    def get_value(self, phoneme: MultiPhoneme,
                  other_phoneme: MultiPhoneme) -> SomeNumber:
        if not isinstance(phoneme, tuple) \
            and not isinstance(phoneme, str) \
            or not isinstance(other_phoneme, tuple) \
            and not isinstance(other_phoneme, str):
            raise TypeError("PhonemeMap keys must be strings or tuples of strings.")
        return self.internal_map.get_value(phoneme, other_phoneme)

    def get_value_or_default(self, phoneme: MultiPhoneme,
                             other_phoneme: MultiPhoneme,
                             default) -> SomeNumber:
        return self.internal_map.get_value_or_default(phoneme, other_phoneme, default)

    def set_value(self, phoneme: Phoneme,
                  other_phoneme: Phoneme,
                  value: SomeNumber) -> None:
        self.internal_map.set_value(phoneme, other_phoneme, value)

    def has_value(self, phoneme: Phoneme, other_phoneme: Phoneme) -> bool:
        return self.internal_map.has_value(phoneme, other_phoneme)

    def is_empty(self):
        return self.internal_map.is_empty()

    def get_key_pairs(self):
        return self.internal_map.get_key_pairs()

    def get_primary_keys(self):
        return self.internal_map.get_primary_keys()

    def get_secondary_keys(self, primary_key: MultiPhoneme):
        return self.internal_map.get_secondary_keys(primary_key)

    # equals
    def __eq__(self, other):
        if isinstance(other, PhonemeMap):
            return self.internal_map == other.internal_map
        if isinstance(other, dict):
            return self.internal_map == other
        return False

    def __str__(self):
        return str(self.internal_map)

    def __repr__(self):
        return str(self.internal_map)

    def __len__(self):
        return self.internal_map.length()
