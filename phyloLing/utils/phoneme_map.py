from copy import deepcopy
from math import inf
from statistics import mean
from typing import Iterable, NewType

Phoneme = NewType('Phoneme', str)
type MultiPhoneme = Phoneme | tuple[Phoneme, ...]
type SomeNumber = int | float

class DoubleMap[TKey, TValue]:
    def __init__(self, default_value: TValue, values: dict=None):
        self.default_value = default_value
        self.values: dict[TKey, dict[TKey, TValue]] = values if values else {}

    def get_value(self, primary_key: TKey, secondary_key: TKey) -> TValue:
        first_map = self.values.get(primary_key)
        if first_map is None:
            self.set_value(primary_key, secondary_key, self.default_value)
            return self.default_value
        second_map_value = first_map.get(secondary_key)
        if second_map_value is None:
            self.set_value(primary_key, secondary_key, self.default_value)
            return self.default_value
        return second_map_value

    def get_value_or_default(self,
                             primary_key: TKey,
                             secondary_key: TKey,
                             default) -> TValue:
        first_map = self.values.get(primary_key)
        if first_map is None:
            return default
        return first_map.get(secondary_key, default)

    def get_primary_key_map(self, primary_key: TKey) -> dict[TKey, TValue]:
        return self.values.get(primary_key)

    def set_value(self, primary_key: TKey, secondary_key: TKey, value: TValue):
        if primary_key not in self.values:
            self.values[primary_key] = {}
        self.values[primary_key][secondary_key] = value
    
    def increment_value(self, primary_key: TKey, secondary_key: TKey, value: SomeNumber):
        if primary_key not in self.values:
            self.values[primary_key] = {}
        if secondary_key not in self.values[primary_key]:
            self.values[primary_key][secondary_key] = self.default_value
        self.values[primary_key][secondary_key] += value
    
    def delete_value(self, primary_key: TKey, secondary_key: TKey):
        if primary_key in self.values:
            if secondary_key in self.values[primary_key]:
                del self.values[primary_key][secondary_key]

    def delete_primary_key(self, primary_key: TKey):
        if primary_key in self.values:
            del self.values[primary_key]

    def is_empty(self):
        return len(self.values) == 0

    def has_value(self, primary_key: TKey,
                  secondary_key: TKey) -> bool:
        return primary_key in self.values and secondary_key in self.values[primary_key]

    def get_key_pairs(self):
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


class PhonemeMap:
    def __init__(self, default_value: SomeNumber = 0, values: dict=None):
        self.default_value: SomeNumber = default_value
        self.internal_map = DoubleMap(default_value, values)
        self.phon_dist_added = False

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

    def get_primary_key_map(self, phoneme: MultiPhoneme) -> dict[MultiPhoneme, SomeNumber]:
        return self.internal_map.get_primary_key_map(phoneme)

    def set_value(self,
                  phoneme: Phoneme,
                  other_phoneme: Phoneme,
                  value: SomeNumber) -> None:
        self.internal_map.set_value(phoneme, other_phoneme, value)

    def increment_value(self,
                        primary_key: Phoneme,
                        secondary_key: Phoneme,
                        value: SomeNumber) -> None:
        self.internal_map.increment_value(primary_key, secondary_key, value)

    def delete_value(self, primary_key: Phoneme, secondary_key: Phoneme):
        self.internal_map.delete_value(primary_key, secondary_key)

    def delete_primary_key(self, primary_key: Phoneme):
        self.internal_map.delete_primary_key(primary_key)

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

    def copy(self):
        return deepcopy(self)
    
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


def average_corrs(corr_dict1: PhonemeMap, corr_dict2: PhonemeMap) -> PhonemeMap:
    avg_corr: PhonemeMap = PhonemeMap(0)
    for (seg1, seg2) in corr_dict1.get_key_pairs():
        avg_corr.set_value(
            seg1,
            seg2,
            mean([corr_dict1.get_value(seg1, seg2), corr_dict2.get_value(seg2, seg1)])
        )
    for (seg2, seg1) in corr_dict2.get_key_pairs():
        if not avg_corr.has_value(seg1, seg2):
            avg_corr.set_value(
                seg1,
                seg2,
                mean([corr_dict1.get_value(seg1, seg2), corr_dict2.get_value(seg2, seg1)])
            )
    return avg_corr


def average_nested_dicts(dict_list: Iterable[PhonemeMap], default=0, drop_inf=True) -> PhonemeMap:
    corr1_all = set(corr1 for d in dict_list for corr1 in d.get_primary_keys())
    corr2_all = {
        corr1: set(
            corr2 for d in dict_list
            for corr2 in d.get_secondary_keys(corr1)
        )
        for corr1 in corr1_all
    }
    results = PhonemeMap(0)
    for corr1 in corr1_all:
        for corr2 in corr2_all[corr1]:
            vals = []
            for d in dict_list:
                value = d.get_value_or_default(corr1, corr2, default)
                if drop_inf and value not in {-inf, inf}:
                    vals.append(value)
                elif not drop_inf:
                    vals.append(value)
            if len(vals) > 0:
                results.set_value(corr1, corr2, mean(vals))
    return results


def reverse_corr_dict_map(corr_dict: PhonemeMap) -> PhonemeMap:
    if not isinstance(corr_dict, PhonemeMap):
        raise ValueError("corr_dict must be a PhonemeMap object")
    reverse = PhonemeMap(0)
    for (seg1, seg2) in corr_dict.get_key_pairs():
        reverse.set_value(seg2, seg1, corr_dict.get_value(seg1, seg2))
    return reverse
