//! A map stored as a vector of entries sorted by key.

use std::borrow::Borrow;
use std::ops::Deref;

/// A map stored as a `Vec` of `(key, value)` entries in ascending key order,
/// with unique keys.
///
/// Phase results keyed by sparse identities such as `NodeId` use this map:
/// its entry order depends only on its contents, so it hashes and compares
/// deterministically inside Salsa structs, and later phases look entries up
/// by binary search in the stored result instead of rebuilding a hash map.
///
/// It dereferences to the sorted entry slice.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct SortedMap<K, V> {
    // The entries carry whatever lifetime `K` and `V` carry.
    #[salsa_value(unsafe(prove(K: salsa::SalsaValue, V: salsa::SalsaValue)))]
    entries: Vec<(K, V)>,
}

impl<K, V> Default for SortedMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<K: Ord, V> SortedMap<K, V> {
    /// The value stored for `key`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.position(key).map(|index| &self.entries[index].1)
    }

    /// Whether the map holds an entry for `key`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.position(key).is_some()
    }

    /// The mutable value stored for `key`.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.position(key).map(|index| &mut self.entries[index].1)
    }

    /// Insert `value` for `key`, returning the value it replaces. Inserting
    /// shifts the later entries; collect a whole map with `FromIterator`.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.entries.binary_search_by(|(entry, _)| entry.cmp(&key)) {
            Ok(index) => Some(std::mem::replace(&mut self.entries[index].1, value)),
            Err(index) => {
                self.entries.insert(index, (key, value));
                None
            }
        }
    }

    /// Remove the entry for `key`, returning its value.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.position(key).map(|index| self.entries.remove(index).1)
    }

    fn position<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.entries
            .binary_search_by(|(entry, _)| entry.borrow().cmp(key))
            .ok()
    }
}

impl<K, V> SortedMap<K, V> {
    /// The keys in ascending order.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> {
        self.entries.iter().map(|(key, _)| key)
    }

    /// The values in ascending key order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.entries.iter().map(|(_, value)| value)
    }

    /// The mutable values in ascending key order.
    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut V> {
        self.entries.iter_mut().map(|(_, value)| value)
    }

    /// The entries in ascending key order, with mutable values. Keys stay
    /// immutable so the order holds.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (&K, &mut V)> {
        self.entries.iter_mut().map(|(key, value)| (&*key, value))
    }

    /// The entries in ascending key order.
    pub fn into_vec(self) -> Vec<(K, V)> {
        self.entries
    }
}

impl<K, V> Deref for SortedMap<K, V> {
    type Target = [(K, V)];

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

/// Collects entries in any order. When a key repeats, the last entry wins,
/// as with `HashMap::extend`.
impl<K: Ord, V> FromIterator<(K, V)> for SortedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut entries: Vec<(K, V)> = iter.into_iter().collect();
        // A stable sort keeps repeated keys in arrival order; keep the last.
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));
        entries.reverse();
        entries.dedup_by(|(later, _), (earlier, _)| later == earlier);
        entries.reverse();
        Self { entries }
    }
}

impl<K, V> IntoIterator for SortedMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a SortedMap<K, V> {
    type Item = &'a (K, V);
    type IntoIter = std::slice::Iter<'a, (K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    /// An edit applied to both the map under test and the `BTreeMap` model.
    /// A small key space makes hits and misses both common.
    #[derive(Clone, Debug)]
    enum Edit {
        Insert(u8, u32),
        Remove(u8),
        AddTo(u8, u32),
    }

    fn edit() -> impl Strategy<Value = Edit> {
        let key = 0u8..16;
        prop_oneof![
            (key.clone(), any::<u32>()).prop_map(|(key, value)| Edit::Insert(key, value)),
            key.clone().prop_map(Edit::Remove),
            (key, any::<u32>()).prop_map(|(key, delta)| Edit::AddTo(key, delta)),
        ]
    }

    /// Check every read accessor against the model.
    fn assert_matches_model(
        map: &SortedMap<u8, u32>,
        model: &BTreeMap<u8, u32>,
    ) -> Result<(), TestCaseError> {
        let expected: Vec<(u8, u32)> = model.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(&**map, &expected[..]);
        prop_assert!(map.keys().copied().eq(model.keys().copied()));
        prop_assert!(map.values().copied().eq(model.values().copied()));
        for key in 0u8..16 {
            prop_assert_eq!(map.get(&key), model.get(&key));
            prop_assert_eq!(map.contains_key(&key), model.contains_key(&key));
        }
        Ok(())
    }

    proptest! {
        /// Any sequence of edits leaves the map equal to a `BTreeMap` given the
        /// same edits, with the same return values along the way.
        #[test]
        fn edits_match_btree_model(
            initial in prop::collection::vec((0u8..16, any::<u32>()), 0..16),
            edits in prop::collection::vec(edit(), 0..64),
        ) {
            let mut map: SortedMap<u8, u32> = initial.iter().copied().collect();
            // `BTreeMap` collection also lets the last repeated key win.
            let mut model: BTreeMap<u8, u32> = initial.iter().copied().collect();
            assert_matches_model(&map, &model)?;

            for edit in edits {
                match edit {
                    Edit::Insert(key, value) => {
                        prop_assert_eq!(map.insert(key, value), model.insert(key, value));
                    }
                    Edit::Remove(key) => {
                        prop_assert_eq!(map.remove(&key), model.remove(&key));
                    }
                    Edit::AddTo(key, delta) => {
                        let actual = map.get_mut(&key).map(|value| {
                            *value = value.wrapping_add(delta);
                            *value
                        });
                        let expected = model.get_mut(&key).map(|value| {
                            *value = value.wrapping_add(delta);
                            *value
                        });
                        prop_assert_eq!(actual, expected);
                    }
                }
                assert_matches_model(&map, &model)?;
            }

            for (value, delta) in map.values_mut().zip(1u32..) {
                *value = value.wrapping_add(delta);
            }
            for ((_, value), delta) in model.iter_mut().zip(1u32..) {
                *value = value.wrapping_add(delta);
            }
            assert_matches_model(&map, &model)?;
            for (key, value) in map.iter_mut() {
                *value ^= u32::from(*key);
            }
            for (key, value) in model.iter_mut() {
                *value ^= u32::from(*key);
            }
            assert_matches_model(&map, &model)?;

            let expected: Vec<(u8, u32)> = model.into_iter().collect();
            prop_assert_eq!(map.clone().into_iter().collect::<Vec<_>>(), expected.clone());
            prop_assert_eq!(map.into_vec(), expected);
        }

        /// Collecting any permutation of the same distinct-key entries gives
        /// equal maps, so the contents alone determine equality and hashing.
        #[test]
        fn permutations_collect_to_equal_maps(
            (entries, shuffled) in prop::collection::btree_map(any::<u16>(), any::<u32>(), 0..32)
                .prop_flat_map(|entries| {
                    let entries: Vec<(u16, u32)> = entries.into_iter().collect();
                    (Just(entries.clone()), Just(entries).prop_shuffle())
                }),
        ) {
            let sorted: SortedMap<u16, u32> = entries.into_iter().collect();
            let permuted: SortedMap<u16, u32> = shuffled.into_iter().collect();
            prop_assert_eq!(sorted, permuted);
        }
    }

    /// Documents the duplicate-key rule with the entries in view: the last
    /// entry for a key wins regardless of where it sits in the input.
    #[test]
    fn collects_in_key_order_with_last_entry_winning() {
        let map: SortedMap<u32, &str> = [(3, "c"), (1, "a"), (3, "C"), (2, "b"), (1, "A")]
            .into_iter()
            .collect();
        assert_eq!(&*map, [(1, "A"), (2, "b"), (3, "C")]);
    }
}
