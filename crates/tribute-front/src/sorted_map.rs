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

    #[test]
    fn collects_in_key_order_with_last_entry_winning() {
        let map: SortedMap<u32, &str> = [(3, "c"), (1, "a"), (3, "C"), (2, "b"), (1, "A")]
            .into_iter()
            .collect();
        assert_eq!(&*map, [(1, "A"), (2, "b"), (3, "C")]);
        assert_eq!(map.get(&3), Some(&"C"));
        assert_eq!(map.get(&4), None);
        assert!(map.contains_key(&2));
    }

    #[test]
    fn edits_values_in_place() {
        let mut map: SortedMap<u32, u32> = [(1, 10), (2, 20), (3, 30)].into_iter().collect();
        *map.get_mut(&2).unwrap() += 1;
        assert_eq!(map.insert(0, 0), None);
        assert_eq!(map.insert(3, 31), Some(30));
        assert_eq!(map.remove(&0), Some(0));
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.remove(&1), None);
        assert_eq!(&*map, [(2, 21), (3, 31)]);
    }

    #[test]
    fn equal_contents_collect_to_equal_maps() {
        let forward: SortedMap<u32, u32> = (0..64).map(|key| (key, key * 2)).collect();
        let backward: SortedMap<u32, u32> = (0..64).rev().map(|key| (key, key * 2)).collect();
        assert_eq!(forward, backward);
    }
}
