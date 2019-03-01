use crate::cell::UnsafeCell;
use crate::ops::Deref;
use crate::mem;
use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering;
use crate::sys_common;
use crate::sys_common::mutex::Mutex;

/// Helper for lazy initialization of a static, with a destructor that runs when the main (Rust)
/// thread exits.
///
/// Currently used only inside the standard library, by the stdio types.
///
/// If there are still child threads around when the main thread exits, they get terminated. But
/// there is a small window where they are not yet terminated and may hold a reference to the
/// contents of our static. To prevent undefined behaviour we track the number of references by
/// handing out `InUseGuard`s, and don't run the destructor if there are any in use.
///
/// # Safety
/// - `UnsafeCell`: We only create a mutable reference during initialization and during the shutdown
///   phase. At both times there can't exist any other references.
/// - Destruction. The `Drop` implementation of `T` should not access references to anything except
///   itself, they are not guaranteed to exist. It should also not rely on other machinery of the
///   standard library to be available.
/// - Initialization. The `init` function for `get` should not call `get` itself, to prevent
///   infinite recursion and acquiring the guard mutex reentrantly.
/// - We use the `Mutex` from `sys::common` because it has a `const` constructor. It currently has
///   UB when acquired reentrantly without calling `init`.
pub struct Lazy<T> {
    guard: Mutex, // Only used to protect initialization.
    in_use_counter: AtomicUsize,
    data: UnsafeCell<Option<T>>,
}

unsafe impl<T> Sync for Lazy<T> {}

const UNINITIALIZED: usize = 0;
const NOT_IN_USE: usize = 1;
const MAX_REFCOUNT: usize = isize::max_value() as usize;
const DESTRUCTOR_RUNNING: usize = usize::max_value();

impl<T> Lazy<T> {
    pub const fn new() -> Lazy<T> {
        Lazy {
            guard: Mutex::new(),
            in_use_counter: AtomicUsize::new(UNINITIALIZED),
            data: UnsafeCell::new(None),
        }
    }
}

impl<T: Send + Sync + 'static> Lazy<T> {
    pub unsafe fn get(&'static self, init: fn() -> T) -> Option<InUseGuard<T>> {
        loop {
            match self.in_use_counter.load(Ordering::Acquire) {
                UNINITIALIZED => {
                    let _guard = self.guard.lock();
                    // Double-check to make sure this `Lazy` didn't get initialized by another
                    // thread in the small window before we acquired the mutex.
                    if self.in_use_counter.load(Ordering::Relaxed) != UNINITIALIZED {
                        return self.get(init);
                    }

                    // Register an `at_exit` handler.
                    let registered = sys_common::at_exit(move || { self.try_drop(); });

                    // Registering the handler will only fail if we are already in the shutdown
                    // phase. In that case don't attempt to initialize.
                    if registered.is_ok() {
                        *self.data.get() = Some(init());
                    }
                    self.in_use_counter.store(NOT_IN_USE + 1, Ordering::Release);
                    break;
                },
                MAX_REFCOUNT..=DESTRUCTOR_RUNNING => {
                    // Protection against overflowing the counter, at the same time catches
                    // `DESTRUCTOR_RUNNING`
                    return None;
                },
                count => {
                    if self.in_use_counter.compare_and_swap(count, count + 1,
                                                            Ordering::Relaxed) == count {
                        break;
                    }
                },
            }
        }

        if let Some(data_ref) = (*self.data.get()).as_ref() {
            return Some(InUseGuard {
                in_use_counter: &self.in_use_counter,
                inner: data_ref,
            });
        }
        // We did not end up returning a reference, decrement the counter again.
        self.in_use_counter.fetch_sub(1, Ordering::Relaxed);
        None
    }

    fn try_drop(&self) -> bool {
        if self.in_use_counter
            .compare_and_swap(NOT_IN_USE, DESTRUCTOR_RUNNING, Ordering::Acquire) == NOT_IN_USE
        {
            let _data = unsafe { mem::replace(&mut *self.data.get(), None) };
            self.in_use_counter.store(NOT_IN_USE, Ordering::Release);
            true
        } else {
            false
        }
    }
}

pub struct InUseGuard<T: 'static> {
    in_use_counter: &'static AtomicUsize,
    inner: &'static T,
}

impl<T> Deref for InUseGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.inner
    }
}

impl<T> Drop for InUseGuard<T> {
    fn drop(&mut self) {
        self.in_use_counter.fetch_sub(1, Ordering::Relaxed);
    }
}
