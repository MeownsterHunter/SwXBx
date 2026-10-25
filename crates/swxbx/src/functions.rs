//! The functions a program can request by name.
//!
//! The list is data, not code: `data/function-names.txt` holds one name per
//! line, sorted. A function's id is its position in that list, so ids are
//! stable for one host interface version.

use std::ffi::{CStr, CString};
use std::sync::OnceLock;

static NAMES: &str = include_str!("../../../data/function-names.txt");

/// Index of a function in the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionId(pub u32);

struct Table {
    names: Vec<&'static str>,
    c_names: Vec<CString>,
}

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| {
        let names: Vec<&'static str> = NAMES.lines().filter(|line| !line.is_empty()).collect();
        let c_names = names
            .iter()
            .map(|name| CString::new(*name).expect("function names contain no NUL"))
            .collect();
        Table { names, c_names }
    })
}

/// Number of functions in the table.
pub fn count() -> usize {
    table().names.len()
}

/// Name of a function, or `None` for an id outside the table.
pub fn name(id: FunctionId) -> Option<&'static str> {
    table().names.get(id.0 as usize).copied()
}

pub(crate) fn c_name(id: FunctionId) -> Option<&'static CStr> {
    table().c_names.get(id.0 as usize).map(CString::as_c_str)
}

/// Id of the function with this exact name.
pub fn lookup(name: &str) -> Option<FunctionId> {
    table()
        .names
        .binary_search(&name)
        .ok()
        .map(|index| FunctionId(index as u32))
}

/// Every function id with its name, in table order.
pub fn all() -> impl Iterator<Item = (FunctionId, &'static str)> {
    table()
        .names
        .iter()
        .enumerate()
        .map(|(index, name)| (FunctionId(index as u32), *name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_is_sorted_and_has_no_duplicates() {
        let names: Vec<_> = all().map(|(_, name)| name).collect();
        assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn the_list_has_the_recorded_number_of_names() {
        // The function table and the public count must remain synchronized.
        assert_eq!(count(), 534);
    }

    #[test]
    fn names_are_plain_identifiers() {
        for (_, name) in all() {
            assert!(name.chars().all(|c| c.is_ascii_alphanumeric()), "{name}");
        }
    }

    #[test]
    fn lookup_finds_every_name_and_nothing_else() {
        for (id, name) in all() {
            assert_eq!(lookup(name), Some(id));
            assert_eq!(super::name(id), Some(name));
        }
        assert_eq!(lookup(""), None);
        assert_eq!(lookup("notAFunction"), None);
        assert_eq!(name(FunctionId(count() as u32)), None);
    }
}
