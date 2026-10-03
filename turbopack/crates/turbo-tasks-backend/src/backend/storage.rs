use std::{
    cell::Cell,
    fmt::{Display, Formatter},
    hash::{BuildHasher, Hash},
    ops::{Deref, DerefMut},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use anyhow::{Context, Result};
use crossbeam_utils::CachePadded;
use hashbrown::hash_table;
use smallvec::SmallVec;
use thread_local::ThreadLocal;
use tracing::span::Id;
use turbo_bincode::{TurboBincodeBuffer, new_turbo_bincode_decoder, new_turbo_bincode_encoder};
use turbo_tasks::{FxDashMap, TaskId, backend::CachedTaskTypeArc, event::Event, parallel};

use crate::{
    backend::storage_schema::{
        DropPartialOutcome, KeyEvictability, TaskStorage, UnevictableReason, ValueEvictability,
    },
    backing_storage::{SnapshotItem, compute_task_type_hash},
    database::key_value_database::KeySpace,
    utils::{
        dash_map_drop_contents::drop_contents,
        dash_map_entry::{TryLockAndRemove, try_lock_and_remove},
        dash_map_multi::{RefMut, get_disjoint_mut},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskDataCategory {
    Meta,
    Data,
    All,
}
impl PartialOrd for TaskDataCategory {
    /// `All` is greater than both `Meta` and `Data`; `Meta` and `Data` are unordered.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;

        use TaskDataCategory::All;
        match (self, other) {
            _ if self == other => Some(Equal),
            (All, _) => Some(Greater),
            (_, All) => Some(Less),
            _ => None,
        }
    }
}

/// Counts of tasks evicted at each level.
#[derive(Debug, Default)]
pub struct EvictionCounts {
    pub key_evictions: usize,
    pub full: usize,
    pub data_and_meta: usize,
    pub data_only: usize,
    pub meta_only: usize,
    /// Per-reason counts of tasks we considered but could not evict, indexed by
    /// `UnevictableReason::index()`.
    pub unevictable_reasons: [usize; UnevictableReason::COUNT],
}

impl std::ops::AddAssign for EvictionCounts {
    fn add_assign(&mut self, rhs: Self) {
        self.key_evictions += rhs.key_evictions;
        self.full += rhs.full;
        self.data_and_meta += rhs.data_and_meta;
        self.data_only += rhs.data_only;
        self.meta_only += rhs.meta_only;
        for i in 0..UnevictableReason::COUNT {
            self.unevictable_reasons[i] += rhs.unevictable_reasons[i];
        }
    }
}

impl Display for EvictionCounts {
    /// Compact `field=value,...` form used as a single tracing span field so that
    /// adding a new counter or `UnevictableReason` variant doesn't require updating
    /// the span field list.
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let skipped: usize = self.unevictable_reasons.iter().sum();
        write!(
            f,
            "task_cache_evictions={},full={},data_and_meta={},data_only={},meta_only={},skipped={}",
            self.key_evictions,
            self.full,
            self.data_and_meta,
            self.data_only,
            self.meta_only,
            skipped,
        )?;
        for reason in UnevictableReason::ALL {
            write!(
                f,
                ",{}={}",
                reason.span_name(),
                self.unevictable_reasons[reason.index()],
            )?;
        }
        Ok(())
    }
}

impl TaskDataCategory {
    pub fn includes_data(self) -> bool {
        matches!(self, TaskDataCategory::Data | TaskDataCategory::All)
    }

    pub fn includes_meta(self) -> bool {
        matches!(self, TaskDataCategory::Meta | TaskDataCategory::All)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecificTaskDataCategory {
    Meta,
    Data,
}

impl From<SpecificTaskDataCategory> for TaskDataCategory {
    fn from(category: SpecificTaskDataCategory) -> Self {
        match category {
            SpecificTaskDataCategory::Meta => TaskDataCategory::Meta,
            SpecificTaskDataCategory::Data => TaskDataCategory::Data,
        }
    }
}

impl SpecificTaskDataCategory {
    /// Returns the KeySpace for storing data of this category
    pub fn key_space(self) -> KeySpace {
        match self {
            SpecificTaskDataCategory::Meta => KeySpace::TaskMeta,
            SpecificTaskDataCategory::Data => KeySpace::TaskData,
        }
    }
}

/// Records exactly what a `track_modification` call changed, so that
/// [`StorageWriteGuard::undo_track_modification`] can reverse it precisely when the mutation it
/// guarded turns out to be a no-op.  This allows us to track modifications 'optimistically' and
/// undo it if the modification turned out to be a no op.  Useful when dealing with datastructures
/// like `AutoSet` that can efficiently say whether or not they were modified.
#[must_use = "a no-op mutation must undo its TrackOutcome; dropping it leaks an over-track"]
pub enum TrackOutcome {
    /// Nothing was tracked: the category was already modified. Undo is a no-op.
    NoChange,
    /// `modified(category)` was set. `bumped` is true if this call also incremented the per-shard
    /// modified counter (i.e. the task had no prior modifications). `inserted_snapshot` is true if
    /// this call also inserted the task's pre-mutation encoded state into the `snapshots` map
    /// (the task was captured by the in-progress snapshot and not persisted yet).
    Tracked {
        category: SpecificTaskDataCategory,
        bumped: bool,
        inserted_snapshot: bool,
    },
}

/// The categories of a task that a snapshot persists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SnapshotMask {
    pub meta: bool,
    pub data: bool,
    /// The task was created since it was last persisted, so it needs a task cache entry.
    pub new_task: bool,
}

impl SnapshotMask {
    /// The categories captured by the in-progress snapshot that are not persisted yet.
    fn pending(task: &TaskStorage) -> Self {
        Self {
            meta: task.flags.meta_snapshot_pending(),
            data: task.flags.data_snapshot_pending(),
            new_task: task.flags.new_task(),
        }
    }

    /// The task's unpersisted modifications (used in drain mode, where nothing is captured
    /// because the map is discarded right after the snapshot).
    fn modified(task: &TaskStorage) -> Self {
        Self {
            meta: task.flags.meta_modified(),
            data: task.flags.data_modified(),
            new_task: task.flags.new_task(),
        }
    }
}

/// Encodes task data, using the provided buffer as a scratch space.  Returns a new exactly sized
/// buffer.
/// This allows reusing the buffer across multiple encode calls to optimize allocations and
/// resulting buffer sizes.
///
/// TODO: The `Result` return type is an artifact of the bincode `Encode` trait requiring
/// fallible encoding. In practice, encoding to a `SmallVec` is infallible (no I/O), and the only
/// real failure mode — a `TypedSharedReference` whose value type has no bincode impl — is a
/// programmer error caught by the panic in the caller. Consider making the bincode encoding trait
/// infallible (i.e. returning `()` instead of `Result<(), EncodeError>`) to eliminate the
/// spurious `Result` threading throughout the encode path.
pub(crate) fn encode_task_contents(
    task: TaskId,
    data: &TaskStorage,
    category: SpecificTaskDataCategory,
    scratch_buffer: &mut TurboBincodeBuffer,
) -> Result<TurboBincodeBuffer> {
    scratch_buffer.clear();
    let mut encoder = new_turbo_bincode_encoder(scratch_buffer);
    data.encode(category, &mut encoder)?;

    if cfg!(feature = "verify_serialization") {
        TaskStorage::new()
            .decode(
                category,
                &mut new_turbo_bincode_decoder(&scratch_buffer[..]),
            )
            .with_context(|| {
                format!(
                    "expected to be able to decode serialized data for '{category:?}' information \
                     for {task}"
                )
            })?;
    }
    Ok(SmallVec::from_slice(scratch_buffer))
}

/// Converts a task's current state into the [`SnapshotItem`] that persistence writes for it.
///
/// Only the categories in `mask` are encoded. A `new_task` additionally carries its task type hash
/// so it can be added to the task cache. When `gc_enabled` is set, a GC-deleted task becomes a
/// [`SnapshotItem::Delete`] tombstone.
///
/// This is shared by the regular snapshot path and by [`StorageWriteGuard::track_modification`],
/// which encodes a task eagerly when it is about to be mutated while a snapshot that captured it
/// has not persisted it yet.
pub(crate) fn encode_snapshot_item(
    task_id: TaskId,
    inner: &TaskStorage,
    mask: SnapshotMask,
    gc_enabled: bool,
    buffer: &mut TurboBincodeBuffer,
) -> Result<SnapshotItem> {
    if task_id.is_transient() {
        unreachable!("transient task_ids should never be enqueued to be persisted");
    }

    if gc_enabled {
        if inner.flags.deleted() {
            debug_assert!(
                !mask.new_task,
                "a scanned GC-deleted task must be persisted; new tasks are discarded by GC"
            );
            let task_type_hash = compute_task_type_hash(
                inner
                    .get_persistent_task_type()
                    .expect("a GC-deleted task must have a task type"),
            );
            return Ok(SnapshotItem::Delete {
                task_id,
                task_type_hash,
            });
        } else {
            debug_assert!(
                !inner.gc_collectible(),
                "tasks scheduled for persistent must not be collectible, this implies a missed \
                 task during GC"
            );
        }
    } else {
        debug_assert!(
            !inner.flags.deleted(),
            "Deleted flags should only be set by GC and it is disabled"
        )
    }

    let meta = if mask.meta {
        Some(
            encode_task_contents(task_id, inner, SpecificTaskDataCategory::Meta, buffer)
                .context("failed to encode task meta data")?,
        )
    } else {
        None
    };

    let data = if mask.data {
        Some(
            encode_task_contents(task_id, inner, SpecificTaskDataCategory::Data, buffer)
                .context("failed to encode task data")?,
        )
    } else {
        None
    };

    let task_type_hash = if mask.new_task {
        let task_type = inner.get_persistent_task_type().expect(
            "It is not possible for a new_task to not have a persistent_task_type.  Task creation \
             for persistent tasks uses a single ExecutionContextImpl for creating the task (which \
             sets new_task) and connect_child (which sets persistent_task_type) and take_snapshot \
             waits for all operations to complete before we start snapshotting.  So task creation \
             will always set the task_type.",
        );
        Some(compute_task_type_hash(task_type))
    } else {
        None
    };

    Ok(SnapshotItem::Put {
        task_id,
        meta,
        data,
        task_type_hash,
    })
}

pub struct Storage {
    snapshot_mode: AtomicBool,
    /// Whether GC is enabled. Used by [`encode_snapshot_item`] when a task has to be encoded
    /// eagerly while a snapshot is in progress.
    gc_enabled: bool,
    /// Per-shard counts of tasks with modified flags set. Incremented when a task
    /// transitions from unmodified to modified. Reset to zero when `take_snapshot` captures the
    /// shard's modified tasks (clearing their modified flags), and re-incremented if a captured
    /// task is not persisted after all (see `SnapshotShard`'s `Drop`). Used to skip unmodified
    /// shards in `take_snapshot`, avoiding unnecessary iteration and enabling early returns
    ///
    /// Indexed by `map.determine_shard(map.hash_usize(&key))` and guaranteed by construction so
    /// that  `shard_modified_counts.len()==map.shards().len()`
    ///
    /// Should only be modified while holding the corresponding dashmap shard lock.
    shard_modified_counts: Box<[CachePadded<AtomicU64>]>,
    /// Copy-on-write snapshots of tasks that were captured by the in-progress snapshot (they have
    /// `*_snapshot_pending` flags) and then modified before the snapshot iterator persisted them.
    /// Each entry holds the captured categories' pre-mutation state, already bincode-encoded into
    /// the [`SnapshotItem`] that persistence writes. Encoding (rather than cloning the
    /// `TaskStorage`) is required for consistency: a clone would share cell contents with
    /// interior mutability via `Arc`, so later mutations could leak into the supposedly frozen
    /// copy. Persistence uses the item as-is, so no work is wasted. Entries are removed when the
    /// iterator persists the task (or when a captured task is not persisted, see
    /// `SnapshotShard`'s `Drop`), so the map is empty outside of snapshots.
    ///
    /// Lock Ordering: `snapshots` locks are acquired **after** `map` locks (see the comment on
    /// `map` below). Holding a `snapshots` shard write lock and then trying to take a `map` shard
    /// write lock is forbidden — it would deadlock against `track_modification_internal` /
    /// `SnapshotShardIter::next`, which take map first.
    snapshots: FxDashMap<TaskId, Box<SnapshotItem>>,
    /// The main storage map
    ///
    /// Lock Ordering: Task creation acquires a `task_cache` lock and then inserts into this map.
    /// Because both datastructures are sharded on different keys, the locks are not 'strictly'
    /// ordered but we should treat them as such
    /// Acquiring locks in the opposite order should be defensive
    ///
    /// Lock Ordering vs. `snapshots`: `map` locks are acquired **before** `snapshots` locks.
    /// `track_modification_internal`, `SnapshotShardIter::next` and `SnapshotShard`'s `Drop` all
    /// hold a `map` shard write lock (via `StorageWriteGuard` / `map.get_mut`) and then take a
    /// `snapshots` shard lock.
    map: FxDashMap<TaskId, Box<TaskStorage>>,
    /// A shared event notified whenever any task finishes restoring (successfully or not).
    ///
    /// Threads waiting for another thread's in-progress restore subscribe to this event,
    /// then re-check the specific task's `restoring`/`restored` bits after waking.
    pub(crate) restored: Event,
    /// Maps `CachedTaskType` → `TaskId` for deduplication of persistent task creation.
    /// This is backed by the TaskCache table in the database.
    ///
    /// LockOrdering: See the comments on [map].
    pub task_cache: FxDashMap<CachedTaskTypeArc, TaskId>,
}

impl Storage {
    pub fn new(shard_amount: usize, small_preallocation: bool, gc_enabled: bool) -> Self {
        let map_capacity: usize = if small_preallocation {
            1024
        } else {
            1024 * 1024
        };

        let map = FxDashMap::with_capacity_and_hasher_and_shard_amount(
            map_capacity,
            Default::default(),
            shard_amount,
        );
        let shard_modified_counts = (0..shard_amount)
            .map(|_| CachePadded::new(AtomicU64::new(0)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            snapshot_mode: AtomicBool::new(false),
            gc_enabled,
            shard_modified_counts,
            snapshots: FxDashMap::with_capacity_and_hasher_and_shard_amount(
                // We expect very few updates to this map since it will only happen when updates
                // race with snapshots.  This never happens in a build and only rarely happens in
                // dev sessions
                0,
                Default::default(),
                shard_amount,
            ),
            map,
            restored: Event::new(|| || "Storage::restored".to_string()),
            task_cache: FxDashMap::default(),
        }
    }

    /// Returns the shard index for the given key in the `map` DashMap.
    fn shard_index(&self, key: &TaskId) -> usize {
        let hash = self.map.hash_usize(key);
        self.map.determine_shard(hash)
    }

    /// Mark a newly allocated task as restored (skip DB queries) and new (include in persistence
    /// snapshots). Optionally sets the `persistent_task_type` eagerly so it's available for
    /// persistence snapshots without needing to propagate it through `connect_child`.
    pub fn initialize_new_task(&self, task_id: TaskId, task_type: Option<CachedTaskTypeArc>) {
        let mut task = self.access_mut(task_id);
        task.flags.set_restored(TaskDataCategory::All);
        task.flags.set_new_task(true);
        task.gc_pin_for_construction();
        if let Some(task_type) = task_type {
            task.set_persistent_task_type(task_type);
            if !task_id.is_transient() {
                // Unconditional track: a new task's type is always a real persistable change.
                let _ =
                    task.track_modification(SpecificTaskDataCategory::Data, "persistent_task_type");
            }
        }
    }

    /// Captures every modified task and returns iterators that persist them. Ends snapshot mode
    /// when the returned `SnapshotGuard` (held by each shard) is dropped.
    ///
    /// **Must be called while operations are excluded** (inside the snapshot phase), so that the
    /// capture is consistent: for each modified task it records the modified categories as
    /// `*_snapshot_pending` and clears the live `modified` flags. Modifications after the
    /// exclusion then land on the task as normal modifications for the next snapshot, and only
    /// captured tasks that are modified before they are persisted need a copy-on-write snapshot
    /// (see `Storage::snapshots`). Encoding happens later, in the returned iterators.
    ///
    /// `process` is called while holding a lock on the task storage, so it can access the
    /// TaskStorage directly without cloning. It receives the categories to encode and a mutable
    /// scratch buffer that can be reused across iterations to avoid repeated allocations.
    ///
    /// The returned shards implement `IntoIterator`. Empty shards (no modified entries) are
    /// filtered out. Captured tasks that are not yielded (e.g. because a shard or its iterator is
    /// dropped early) are marked as modified again.
    ///
    /// When `drain_entries` is true (shutdown only), the scan drains the map: unmodified entries
    /// are erased and freed immediately, and the modified entries are moved out into the
    /// returned shard iterators, which free each task's memory as it is serialized rather than
    /// after the whole batch is written.
    pub fn take_snapshot<
        'l,
        P: for<'a> Fn(
                TaskId,
                &'a TaskStorage,
                SnapshotMask,
                &mut TurboBincodeBuffer,
            ) -> SnapshotItem
            + Sync,
    >(
        &'l self,
        guard: SnapshotGuard<'l>,
        process: &'l P,
        drain_entries: bool,
    ) -> Vec<SnapshotShard<'l, P>> {
        let guard = Arc::new(guard);

        let shards: Vec<_> = self.map.shards().iter().enumerate().collect();

        // The number of shards is much larger than the number of threads, so the effect of the
        // locks held is negligible.
        parallel::map_collect::<_, _, Vec<_>>(&shards, |&(shard_idx, shard)| {
            // Check how many modifications there are in this shard. Operations are excluded, so
            // there are no racing writes and we can reset the count: every modified task in the
            // shard is captured (and its modified flags cleared) below.
            let modified_count = self.shard_modified_counts[shard_idx].swap(0, Ordering::Relaxed);

            if modified_count == 0 && !drain_entries {
                // Nothing to persist in this shard and we're keeping the map, so skip the scan.
                // TODO: when not draining but eviction is enabled we should run that logic here as
                // well
                return None;
            }

            // Scan the shard once, building the work this shard's iterator will perform. The two
            // modes carry different data so that `next` has no per-item `drain` branch:
            // - keep mode collects the modified `TaskId`s and looks them up again while iterating.
            // - drain mode erases the unmodified entries here and then moves the remaining
            //   (modified-only) table out of the map, so the iterator owns and drains it directly.
            let work = {
                let mut shard_guard = shard.write();
                if drain_entries {
                    shard_guard.retain(|(key, task)| {
                        let modified_task = task.flags.any_modified();
                        if modified_task {
                            debug_assert!(
                                !key.is_transient(),
                                "found a modified transient task: {key:?}"
                            );
                        }
                        // Unmodified entries are not part of the snapshot. Remove and free them
                        // now so the table we move out below holds only modified entries.
                        modified_task
                    });
                    if shard_guard.is_empty() {
                        // The shard held only unmodified entries, which we've now erased and freed.
                        // No iterator is created for an empty shard.
                        return None;
                    }
                    // Move the modified-only table out of the map. Iterating it frees each task box
                    // as it is serialized, and the shard's table allocation is released here.
                    ShardWork::Drain(std::mem::take(&mut *shard_guard).into_iter())
                } else {
                    let mut modified = Vec::with_capacity(modified_count as usize);
                    for (key, task) in shard_guard.iter_mut() {
                        // Only check modified flags — transient tasks never have modified flags set
                        // (track_modification guards against it), so this naturally excludes them.
                        // new_task always comes with modified flags (set_persistent_task_type calls
                        // track_modification), so any_modified() is sufficient.
                        if task.flags.any_modified() {
                            debug_assert!(
                                !key.is_transient(),
                                "found a modified transient task: {key:?}"
                            );
                            debug_assert!(!task.flags.any_snapshot_pending());
                            // Capture: move the modified categories to `*_snapshot_pending`.
                            // `new_task` stays set until the task is persisted; it can't be set
                            // again on an existing task, so it still reflects the capture.
                            let flags = &mut task.flags;
                            flags.set_meta_snapshot_pending(flags.meta_modified());
                            flags.set_data_snapshot_pending(flags.data_modified());
                            flags.set_meta_modified(false);
                            flags.set_data_modified(false);
                            modified.push(*key);
                        }
                    }
                    // modified_count > 0 (we returned early otherwise), so this is never empty.
                    debug_assert!(!modified.is_empty());
                    ShardWork::Keep(modified)
                }
            };

            Some(SnapshotShard {
                shard_idx,
                work,
                storage: self,
                process,
                _guard: guard.clone(),
            })
        })
        .into_iter()
        .flatten()
        .collect()
    }

    /// Enter snapshot mode and return a guard that will call `end_snapshot` on drop.
    ///
    /// Returns whether any shard has modifications. Per-shard counts are reset in
    /// `take_snapshot` as each shard is captured.
    ///
    /// Safety invariant: `start_snapshot` and `end_snapshot` are always called
    /// sequentially within a single `snapshot_and_persist` invocation (the sole
    /// caller). There is no concurrent snapshot lifecycle, so they cannot race.
    pub fn start_snapshot(&self) -> (SnapshotGuard<'_>, bool) {
        self.snapshot_mode.store(true, Ordering::Release);
        let has_modifications = self
            .shard_modified_counts
            .iter()
            .any(|c| c.load(Ordering::Relaxed) > 0);
        (SnapshotGuard::new(self), has_modifications)
    }

    /// End snapshot mode.
    ///
    /// Captured tasks are persisted (or restored as modified) by the shard iterators, which also
    /// remove their `snapshots` entries, so there is nothing left to reconcile here.
    fn end_snapshot(&self) {
        self.snapshot_mode.store(false, Ordering::Release);
        debug_assert!(
            self.snapshots.is_empty(),
            "all copy-on-write snapshots must be consumed by the snapshot iterators"
        );
        // If we are saving a non-trivial amount of memory just clear it out.
        if self.snapshots.capacity() > 1024 {
            self.snapshots.shrink_to_fit();
        }
    }

    /// Returns true if actively snapshotting.
    fn snapshot_mode(&self) -> bool {
        self.snapshot_mode.load(Ordering::Acquire)
    }

    pub fn access_mut(&self, key: TaskId) -> StorageWriteGuard<'_> {
        let inner = match self.map.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(e) => e.into_ref(),
            dashmap::mapref::entry::Entry::Vacant(e) => e.insert(Box::new(TaskStorage::new())),
        };
        StorageWriteGuard {
            storage: self,
            inner: inner.into(),
        }
    }

    /// Like [`Self::access_mut`], but keeps the map entry so the caller can still remove it.
    pub fn access_entry_mut(&self, key: TaskId) -> TaskEntryGuard<'_> {
        let entry = match self.map.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(e) => e,
            dashmap::mapref::entry::Entry::Vacant(e) => {
                e.insert_entry(Box::new(TaskStorage::new()))
            }
        };
        TaskEntryGuard {
            storage: self,
            entry,
        }
    }

    /// Read-only access to an already resident task. Returns `None` if the task isnt in memory
    /// resident. The closure runs while a shard read lock is held, so it must be cheap and must
    /// not re-enter the map.
    pub fn with_task<R>(&self, key: TaskId, f: impl FnOnce(&TaskStorage) -> R) -> Option<R> {
        let task = self.map.get(&key)?;
        Some(f(task.value()))
    }

    /// The number of tasks resident in the map.
    #[doc(hidden)]
    pub fn resident_task_count_for_testing(&self) -> usize {
        self.map.len()
    }

    /// The number of shards in the resident map. GC seeds one `ScanShard` job per index; the slice
    /// returned by `map.shards()` is fixed for the map's lifetime, so an index is a stable handle
    /// to one shard.
    pub fn shard_count(&self) -> usize {
        self.map.shards().len()
    }

    /// Scans a **single** shard by index, invoking `on_candidate` for each resident task whose
    /// storage passes [`TaskStorage::gc_collectible`].
    pub fn gc_scan_shard(&self, index: usize, mut on_candidate: impl FnMut(TaskId)) {
        let shard = self.map.shards()[index].read();
        for (task_id, task) in shard.iter() {
            if task.gc_collectible() {
                on_candidate(*task_id);
            }
        }
    }

    /// Return the set of all known live roots.
    pub fn gc_scan_roots(&self) -> impl Iterator<Item = TaskId> {
        let per_shard: Vec<Vec<TaskId>> =
            parallel::map_collect(&(0..self.shard_count()).collect::<Vec<_>>(), |&index| {
                let mut roots = Vec::new();
                let shard = self.map.shards()[index].read();
                for (task_id, task) in shard.iter() {
                    if !task_id.is_transient() && task.gc_is_root() {
                        roots.push(*task_id);
                    }
                }
                roots
            });

        per_shard.into_iter().flatten()
    }

    pub fn access_pair_mut(
        &self,
        key1: TaskId,
        key2: TaskId,
    ) -> (StorageWriteGuard<'_>, StorageWriteGuard<'_>) {
        let (a, b) = get_disjoint_mut(&self.map, key1, key2, || Box::new(TaskStorage::new()));
        (
            StorageWriteGuard {
                storage: self,
                inner: a,
            },
            StorageWriteGuard {
                storage: self,
                inner: b,
            },
        )
    }

    pub fn drop_contents(&self) {
        drop_contents(&self.map);
        drop_contents(&self.snapshots);
    }

    /// Drop the `task_cache` map, freeing its memory.
    pub(crate) fn drop_task_cache(&self) {
        drop_contents(&self.task_cache);
    }

    /// Evict tasks from in-memory storage after a successful snapshot.
    ///
    /// Iterates all tasks and applies the eviction level returned by
    /// `TaskStorage::evictability()`:
    /// - `Full`: remove from map entirely
    /// - `DataAndMeta`: drop both data and meta fields, keep task in map
    /// - `DataOnly`: drop data fields only
    /// - `MetaOnly`: drop meta fields only
    /// - `No`: skip
    ///
    /// Must be called when NOT in snapshot mode (i.e., after `end_snapshot()`).
    pub fn evict_after_snapshot(&self, parent_span: Option<Id>) -> EvictionCounts {
        let span = tracing::trace_span!(
            parent: parent_span,
            "evict_after_snapshot",
            total_task_cache_keys = self.task_cache.len(),
            total_map_keys = self.map.len(),
            counts = tracing::field::Empty,
        )
        .entered();
        debug_assert!(
            !self.snapshot_mode(),
            "evict_after_snapshot must not be called during snapshot mode"
        );

        let counts: Vec<EvictionCounts> = parallel::map_collect(self.map.shards(), |shard| {
            let mut shard = shard.write();
            let mut evicted = EvictionCounts::default();
            // task_cache removals that we couldn't perform inline because the target shard
            // was contended. We defer them until after the map shard lock is released to
            // avoid a lock cycle with get_or_create_persistent_task, which takes task_cache
            // before map. Allocated lazily on first conflict.
            let mut deferred_task_cache_removals: Vec<CachedTaskTypeArc> = Vec::new();
            // Remove a task type from `task_cache`, deferring on contention. Shared by the
            // GC-deleted path below and the ordinary key eviction.
            let remove_from_task_cache =
                |evicted: &mut EvictionCounts,
                 deferred: &mut Vec<CachedTaskTypeArc>,
                 task_type: &CachedTaskTypeArc| {
                    match try_lock_and_remove(&self.task_cache, task_type.as_ref()) {
                        TryLockAndRemove::Removed => {
                            evicted.key_evictions += 1;
                        }
                        TryLockAndRemove::NotFound => {
                            // Generally this should be rare, it more or less implies something
                            // else is concurrently holding the Arc
                        }
                        TryLockAndRemove::WouldBlock => {
                            // Contention, to avoid a deadlock just defer
                            deferred.push(task_type.clone());
                        }
                    }
                };
            shard.retain(|(task_id, task)| {
                // Transient tasks can not be evicted at all, unless they are fully
                // delete by the GC.
                if task_id.is_transient() && !task.flags.deleted() {
                    evicted.unevictable_reasons[UnevictableReason::Transient.index()] += 1;
                    return true;
                }
                // All GC'd tasks were tombstoned during the snapshot (or are not persisted) so we
                // can drop them fully now.
                if task.flags.deleted() {
                    if let Some(task_type) = task.get_persistent_task_type() {
                        remove_from_task_cache(
                            &mut evicted,
                            &mut deferred_task_cache_removals,
                            task_type,
                        );
                    }
                    evicted.full += 1;
                    return false;
                }
                let (key_evictability, value_evictability) = task.evictability();
                match key_evictability {
                    KeyEvictability::Evictable => {
                        // The task type is persisted to backing storage (new_task = false),
                        // so task_cache is a pure perf cache. Remove it now; it will be
                        // re-populated by task_by_type() on the next cache miss.
                        let task_type = task.get_persistent_task_type().unwrap();
                        // Only try to acquire the lock, if we cannot just remove at the end
                        // Because `get_or_create_task` acquires 'task_cache' then `storage.map` and
                        // we do the opposite we need to be defensive here.  Attempting here is just
                        // an optimization to avoid pushing into `deferred_task_cache_removals`
                        remove_from_task_cache(
                            &mut evicted,
                            &mut deferred_task_cache_removals,
                            task_type,
                        );
                    }
                    KeyEvictability::AlreadyEvicted | KeyEvictability::Unevictable => {}
                }
                match value_evictability {
                    ValueEvictability::Evictable { meta, data } => {
                        match task.drop_partial(data, meta) {
                            DropPartialOutcome::Empty => {
                                evicted.full += 1;
                                return false;
                            }
                            DropPartialOutcome::HasResidue => {
                                if data && meta {
                                    evicted.data_and_meta += 1;
                                } else if data {
                                    evicted.data_only += 1;
                                } else {
                                    debug_assert!(meta);
                                    evicted.meta_only += 1;
                                }
                            }
                        }
                    }
                    ValueEvictability::Unevictable(reason) => {
                        evicted.unevictable_reasons[reason.index()] += 1;
                    }
                }
                true
            });
            // Shrink the shard if it's less than half full, to reclaim slack capacity
            // after bulk evictions. We already hold the write lock, so this is free
            // from a locking perspective. TaskId hashing is cheap (it's just an integer).
            let len = shard.len();
            if shard.capacity() > len * 2 {
                shard.shrink_to(len, |(k, _v)| self.map.hasher().hash_one(k));
            }
            // Release the map shard lock before draining deferred removals so that a thread
            // holding a task_cache shard lock and waiting on this map shard can make progress.
            drop(shard);
            for task_type in deferred_task_cache_removals {
                if self.task_cache.remove(task_type.as_ref()).is_some() {
                    evicted.key_evictions += 1;
                }
            }
            evicted
        });

        let mut totals = EvictionCounts::default();
        for evicted in counts {
            totals += evicted;
        }
        // Shrink task_cache only when we evicted more entries than remain — i.e. the map
        // is less than half full. Rehashing each surviving CachedTaskType isn't free, so
        // we gate it on meaningful slack. Within that, walk shards in parallel and shrink
        // each one independently if it is itself less than half full.
        if totals.key_evictions > self.task_cache.len() {
            parallel::for_each(self.task_cache.shards(), |shard| {
                let mut shard = shard.write();
                let len = shard.len();
                if shard.capacity() > len * 2 {
                    shard.shrink_to(len, |(k, _v)| self.task_cache.hasher().hash_one(k));
                }
            });
        }
        span.record("counts", tracing::field::display(&totals));

        totals
    }
}

/// A write guard that still owns its map entry, so the task can be removed under the lock that is
/// already held.
///
/// Use [`Storage::access_entry_mut`] to obtain one. Convert it with [`Self::into_write_guard`] once
/// removal is no longer a possibility, or call [`Self::discard`] to drop the entry outright.
pub struct TaskEntryGuard<'a> {
    storage: &'a Storage,
    entry: dashmap::mapref::entry::OccupiedEntry<'a, TaskId, Box<TaskStorage>>,
}

impl<'a> TaskEntryGuard<'a> {
    /// Removes this task's entry.
    pub fn discard(self) {
        self.entry.remove();
    }

    /// Gives up the ability to remove the entry, yielding an ordinary write guard.
    pub fn into_write_guard(self) -> StorageWriteGuard<'a> {
        StorageWriteGuard {
            storage: self.storage,
            inner: self.entry.into_ref().into(),
        }
    }
}

impl Deref for TaskEntryGuard<'_> {
    type Target = TaskStorage;
    fn deref(&self) -> &Self::Target {
        self.entry.get()
    }
}

impl DerefMut for TaskEntryGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.entry.get_mut()
    }
}

pub struct StorageWriteGuard<'a> {
    storage: &'a Storage,
    inner: RefMut<'a, TaskId, Box<TaskStorage>>,
}

impl StorageWriteGuard<'_> {
    /// Tracks mutation of this task.
    #[inline(always)]
    pub fn track_modification(
        &mut self,
        category: SpecificTaskDataCategory,
        #[allow(unused_variables)] name: &str,
    ) -> TrackOutcome {
        debug_assert!(
            !self.inner.key().is_transient(),
            "transient task_ids should never be enqueued to be persisted"
        );
        self.track_modification_internal(
            category,
            #[cfg(feature = "trace_task_modification")]
            name,
        )
    }

    fn track_modification_internal(
        &mut self,
        category: SpecificTaskDataCategory,
        #[cfg(feature = "trace_task_modification")] name: &str,
    ) -> TrackOutcome {
        // Transient tasks are never persisted, so tracking modifications is meaningless.
        // All callers (TaskGuard, initialize_new_task) already
        // guard against this, but we enforce it here as defense-in-depth.
        debug_assert!(
            !self.inner.key().is_transient(),
            "track_modification called on transient task {:?}",
            self.inner.key()
        );
        let flags = &self.inner.flags;
        if flags.is_modified(category) {
            // Already tracked. If the task is captured by the in-progress snapshot, that earlier
            // tracking (which happened after the capture, since the capture clears the modified
            // flags) already froze its snapshot state.
            return TrackOutcome::NoChange;
        }
        #[cfg(feature = "trace_task_modification")]
        let _span = tracing::trace_span!("mark_modified", name).entered();
        // If the in-progress snapshot captured this task and hasn't persisted it yet, freeze the
        // captured categories before this mutation lands (copy-on-write). Only the first
        // modification after the capture needs to do this; later ones find the entry.
        let inserted_snapshot =
            flags.any_snapshot_pending() && !self.storage.snapshots.contains_key(self.inner.key());
        if inserted_snapshot {
            let item = self.encode_for_snapshot();
            self.storage
                .snapshots
                .insert(*self.inner.key(), Box::new(item));
        }
        let bumped = !self.inner.flags.any_modified();
        if bumped {
            let shard_idx = self.storage.shard_index(self.inner.key());
            self.storage.shard_modified_counts[shard_idx].fetch_add(1, Ordering::Relaxed);
        }
        self.inner.flags.set_modified(category, true);
        TrackOutcome::Tracked {
            category,
            bumped,
            inserted_snapshot,
        }
    }

    /// Encodes the task's captured, not yet persisted (`*_snapshot_pending`) categories into the
    /// [`SnapshotItem`] the racing persistence will write.
    ///
    /// We encode instead of cloning the `TaskStorage` because cell contents may have interior
    /// mutability and would be shared with a clone, which would break the consistency of the
    /// snapshot. Persistence uses this item directly, so the only cost is the temporary memory.
    /// This is rare, so we use a fresh scratch buffer.
    #[cold]
    fn encode_for_snapshot(&self) -> SnapshotItem {
        let task_id = *self.inner.key();
        let mut buffer = TurboBincodeBuffer::new();
        encode_snapshot_item(
            task_id,
            &self.inner,
            SnapshotMask::pending(&self.inner),
            self.storage.gc_enabled,
            &mut buffer,
        )
        .unwrap_or_else(|err| panic!("Serializing task {task_id} for a snapshot failed: {err:?}"))
    }

    /// Reverse a [`TrackOutcome`] produced by [`Self::track_modification`] when the mutation it
    /// guarded changed nothing persistable.
    ///
    /// # Correctness
    ///
    /// The `outcome` MUST be applied to the **same `StorageWriteGuard`** that produced it, with the
    /// map shard write lock held continuously in between — i.e. `track_modification`, the mutation,
    /// and `undo_track_modification` all run within one guard's lifetime. The guard holds its shard
    /// write lock for its whole lifetime, so this guarantees no other thread observed the tracked
    /// state, and that `bumped` / `inserted_snapshot` still describe reality (the counter and
    /// `snapshots` entry are only mutated under that lock). Because those flags record whether
    /// *this* call created the state, undo never clears a flag, counter, or snapshot entry that a
    /// prior modification owns.
    pub fn undo_track_modification(&mut self, outcome: TrackOutcome) {
        match outcome {
            TrackOutcome::NoChange => {}
            TrackOutcome::Tracked {
                category,
                bumped,
                inserted_snapshot,
            } => {
                self.inner.flags.set_modified(category, false);
                if bumped {
                    let shard_idx = self.storage.shard_index(self.inner.key());
                    self.storage.shard_modified_counts[shard_idx].fetch_sub(1, Ordering::Relaxed);
                }
                if inserted_snapshot {
                    self.storage.snapshots.remove(self.inner.key());
                }
            }
        }
    }

    /// Clears all modified/new flags for a GC-collected task that was **never persisted**
    /// (`new_task`).
    pub fn discard_modifications_for_gc_new_task(&mut self) {
        debug_assert!(
            !self.storage.snapshot_mode(),
            "discard_modifications_for_gc_new_task must run before the snapshot starts"
        );
        debug_assert!(
            self.inner.flags.new_task(),
            "only a never-persisted (new_task) collected task may be discarded this way"
        );
        if self.inner.flags.any_modified() {
            let shard_idx = self.storage.shard_index(self.inner.key());
            self.storage.shard_modified_counts[shard_idx].fetch_sub(1, Ordering::Relaxed);
        }
        self.inner.flags.set_meta_modified(false);
        self.inner.flags.set_data_modified(false);
        self.inner.flags.set_new_task(false);
    }
}

impl Deref for StorageWriteGuard<'_> {
    type Target = TaskStorage;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for StorageWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// How big of a buffer to allocate initially. Based on metrics from a large
/// application this should cover about 98% of values with no resizes.
const SCRATCH_BUFFER_INITIAL_SIZE: usize = 4096;

/// State machine for a per-thread scratch buffer slot.
///
/// Transitions:
/// - `Uninit` → `Taken` (first take)
/// - `Available` → `Taken` (subsequent takes)
/// - `Taken` → `Available` (return)
///
/// Any other transition is a bug (e.g. double-take or double-return).
#[derive(Default)]
enum ScratchBufferSlot {
    /// No buffer has been allocated on this thread yet.
    #[default]
    Uninit,
    /// The buffer is currently checked out.
    Taken,
    /// The buffer is available for reuse.
    Available(TurboBincodeBuffer),
}

pub struct SnapshotGuard<'l> {
    storage: &'l Storage,
    /// Per-thread scratch buffers for encoding task data. Buffers are taken
    /// by `SnapshotShardIter` on creation and returned on drop, allowing reuse
    /// across multiple shards processed by the same thread. When the guard is
    /// dropped (after all iterators are done), the `ThreadLocal` drops too,
    /// freeing all buffers.
    scratch_buffers: ThreadLocal<Cell<ScratchBufferSlot>>,
}

impl<'l> SnapshotGuard<'l> {
    fn new(storage: &'l Storage) -> Self {
        Self {
            storage,
            scratch_buffers: ThreadLocal::new(),
        }
    }

    fn take_scratch_buffer(&self) -> TurboBincodeBuffer {
        let cell = self.scratch_buffers.get_or_default();
        match cell.take() {
            ScratchBufferSlot::Available(buf) => {
                cell.set(ScratchBufferSlot::Taken);
                buf
            }
            ScratchBufferSlot::Uninit => {
                cell.set(ScratchBufferSlot::Taken);
                TurboBincodeBuffer::with_capacity(SCRATCH_BUFFER_INITIAL_SIZE)
            }
            ScratchBufferSlot::Taken => {
                panic!("scratch buffer taken twice without being returned");
            }
        }
    }

    fn return_scratch_buffer(&self, buffer: TurboBincodeBuffer) {
        let cell = self.scratch_buffers.get_or_default();
        match cell.take() {
            ScratchBufferSlot::Taken => cell.set(ScratchBufferSlot::Available(buffer)),
            ScratchBufferSlot::Available(_) => {
                panic!("scratch buffer returned without being taken (already available)");
            }
            ScratchBufferSlot::Uninit => {
                panic!("scratch buffer returned without being taken (uninit)");
            }
        }
    }
}

impl Drop for SnapshotGuard<'_> {
    fn drop(&mut self) {
        self.storage.end_snapshot();
    }
}

/// The work a single shard's iterator performs, with the snapshot mode encoded in the data rather
/// than a runtime flag re-checked per item. Built by `take_snapshot`'s scan.
enum ShardWork {
    /// Normal snapshot: the tasks captured by the scan. Each is looked up in the map while
    /// iterating and persisted from its copy-on-write snapshot (if it was modified since the
    /// capture) or from its live state, then its `*_snapshot_pending` flags are cleared.
    Keep(Vec<TaskId>),
    /// Shutdown drain: the scan already erased the unmodified entries and moved the remaining
    /// (modified-only) shard table out of the map. The iterator owns that table and drains it
    /// directly, freeing each task box as it is serialized. No second map lookup, no flag
    /// bookkeeping (the whole map is discarded right after this snapshot).
    Drain(hash_table::IntoIter<(TaskId, Box<TaskStorage>)>),
}

pub struct SnapshotShard<'l, P> {
    shard_idx: usize,
    work: ShardWork,
    storage: &'l Storage,
    process: &'l P,
    /// Held for its `Drop` impl — ensures snapshot mode ends when all shards are done.
    _guard: Arc<SnapshotGuard<'l>>,
}

impl<'l, P> IntoIterator for SnapshotShard<'l, P>
where
    P: Fn(TaskId, &TaskStorage, SnapshotMask, &mut TurboBincodeBuffer) -> SnapshotItem + Sync,
{
    type Item = SnapshotItem;
    type IntoIter = SnapshotShardIter<'l, P>;

    fn into_iter(self) -> Self::IntoIter {
        let buffer = self._guard.take_scratch_buffer();
        SnapshotShardIter {
            shard: self,
            buffer,
        }
    }
}

impl<P> Drop for SnapshotShard<'_, P> {
    /// Captured tasks that were not persisted (the shard or its iterator was dropped before
    /// yielding them) are marked as modified again so the next snapshot persists them.
    fn drop(&mut self) {
        let ShardWork::Keep(remaining) = &mut self.work else {
            // Drain mode: the map is discarded after this snapshot anyway.
            return;
        };
        for task_id in remaining.drain(..) {
            // Lock order: map first, then snapshots.
            let Some(mut task) = self.storage.map.get_mut(&task_id) else {
                continue;
            };
            self.storage.snapshots.remove(&task_id);
            let already_modified = task.flags.any_modified();
            let flags = &mut task.flags;
            if flags.meta_snapshot_pending() {
                flags.set_meta_modified(true);
            }
            if flags.data_snapshot_pending() {
                flags.set_data_modified(true);
            }
            flags.set_meta_snapshot_pending(false);
            flags.set_data_snapshot_pending(false);
            if !already_modified && task.flags.any_modified() {
                self.storage.shard_modified_counts[self.shard_idx].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Iterator over a single shard's snapshot items. Holds a thread-local scratch
/// buffer for the duration of iteration and returns it on drop.
pub struct SnapshotShardIter<'l, P> {
    shard: SnapshotShard<'l, P>,
    buffer: TurboBincodeBuffer,
}

impl<'l, P> Iterator for SnapshotShardIter<'l, P>
where
    P: Fn(TaskId, &TaskStorage, SnapshotMask, &mut TurboBincodeBuffer) -> SnapshotItem + Sync,
{
    type Item = SnapshotItem;

    fn next(&mut self) -> Option<Self::Item> {
        let process = self.shard.process;
        let buffer = &mut self.buffer;
        match &mut self.shard.work {
            ShardWork::Keep(captured) => {
                // Leave the task in `captured` until encoding succeeds: if `process` panics,
                // dropping the shard still restores the captured flags for this task.
                let task_id = *captured.last()?;
                let storage = self.shard.storage;
                let mut inner = storage.map.get_mut(&task_id).unwrap();
                // If the task was modified since the capture, `track_modification` already encoded
                // its captured state; persist that instead of the (newer) live data.
                let item = match storage.snapshots.remove(&task_id) {
                    Some((_, item)) => *item,
                    None => process(task_id, &inner, SnapshotMask::pending(&inner), buffer),
                };
                // The captured state is persisted. Modifications since the capture (if any) are
                // already tracked by the live `modified` flags.
                inner.flags.set_meta_snapshot_pending(false);
                inner.flags.set_data_snapshot_pending(false);
                inner.flags.set_new_task(false);
                drop(inner);
                captured.pop();
                Some(item)
            }
            ShardWork::Drain(entries) => {
                // Shutdown only: the scan already moved this shard's modified entries out of the
                // map, so we own each `Box<TaskStorage>` here. Serialize from a borrow of the owned
                // box and let it drop at the end of this branch — freeing the task's memory as it
                // is persisted rather than after the whole batch is written. We skip the flag
                // bookkeeping the normal path does, since the entire map is discarded right after
                // this snapshot.
                let (task_id, inner) = entries.next()?;
                Some(process(
                    task_id,
                    &inner,
                    SnapshotMask::modified(&inner),
                    buffer,
                ))
            }
        }
    }
}

impl<P> Drop for SnapshotShardIter<'_, P> {
    fn drop(&mut self) {
        self.shard
            ._guard
            .return_scratch_buffer(std::mem::take(&mut self.buffer));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use turbo_bincode::TurboBincodeBuffer;
    use turbo_tasks::TaskId;

    use super::{
        SnapshotMask, SpecificTaskDataCategory, Storage, TaskStorage, TrackOutcome,
        encode_task_contents,
    };
    use crate::{backing_storage::SnapshotItem, data::OutputValue};

    fn non_transient_task(id: u32) -> TaskId {
        // TRANSIENT_TASK_BIT is 0x2000_0000; any id without that bit is non-transient.
        TaskId::new(id).expect("id must be non-zero")
    }

    #[test]
    fn new_task_is_pinned_during_construction() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        storage.initialize_new_task(task_id, None);

        let task = storage.access_mut(task_id);
        assert_eq!(task.gc_transient_ref_count(), 1);
        assert!(!task.gc_collectible());
    }

    /// A process fn that returns a non-empty SnapshotItem so the iterator doesn't
    /// silently skip items via the "encoding failed" error path.
    fn dummy_process(
        task_id: TaskId,
        _: &super::TaskStorage,
        _: SnapshotMask,
        _: &mut TurboBincodeBuffer,
    ) -> SnapshotItem {
        SnapshotItem::Put {
            task_id,
            meta: Some(TurboBincodeBuffer::default()),
            data: None,
            task_type_hash: None,
        }
    }

    /// Regression test: a task modified before a snapshot and then modified *again* during
    /// snapshot iteration must serialize the captured state and carry the new modification
    /// forward to the next cycle.
    ///
    /// Sequence of events:
    /// 1. Task is modified (data_modified = true) → added to shard_modified_counts.
    /// 2. `start_snapshot` puts us in snapshot mode.
    /// 3. `take_snapshot` captures the task: `data_modified` moves to `data_snapshot_pending`.
    /// 4. **Between capture and iteration**: `track_modification` is called on the same category.
    ///    The task is pending and has no copy-on-write snapshot yet, so its captured state is
    ///    encoded into `snapshots`, and `data_modified` is set again.
    /// 5. `SnapshotShardIter::next` yields the pre-encoded item, removes the snapshots entry, and
    ///    clears the pending flag. `data_modified` stays set for the next cycle.
    // `take_snapshot` uses `parallel::map_collect` which calls `block_in_place` internally,
    // requiring a multi-threaded Tokio runtime.
    #[tokio::test(flavor = "multi_thread")]
    async fn modify_during_snapshot_clears_live_modified_flags() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        // Step 1: modify the task outside snapshot mode (data_modified = true).
        {
            let mut guard = storage.access_mut(task_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        }

        // Step 2: enter snapshot mode.
        let (snapshot_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);

        // Step 3: `take_snapshot` captures the task.
        let shards = storage.take_snapshot(snapshot_guard, &dummy_process, false);
        {
            let guard = storage.access_mut(task_id);
            assert!(!guard.flags.data_modified());
            assert!(guard.flags.data_snapshot_pending());
        }

        // Step 4: now that the capture is done but before we consume the iterator,
        // modify the task again: the captured state is frozen into `snapshots`.
        {
            let mut guard = storage.access_mut(task_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Data, "test");
            assert!(guard.flags.data_modified());
            assert!(storage.snapshots.contains_key(&task_id));
        }

        // Step 5: consume the iterator.
        let items: Vec<_> = shards
            .into_iter()
            .flat_map(|shard| shard.into_iter())
            .collect();

        // The pre-encoded snapshot item should have been returned.
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].task_id(), task_id);

        {
            let guard = storage.access_mut(task_id);
            assert!(guard.flags.data_modified());
            assert!(!guard.flags.any_snapshot_pending());
        }

        // The new modification must be reflected in shard_modified_counts so the next
        // snapshot cycle picks it up. Verify by starting another snapshot.
        let (_guard2, has_modifications) = storage.start_snapshot();
        assert!(
            has_modifications,
            "shard_modified_counts must count the modification made after the capture"
        );
    }

    /// A task modified in one category before a snapshot, then modified in a *different* category
    /// during snapshot iteration, must not panic and must carry the new modification forward.
    ///
    /// Sequence of events:
    /// 1. Task meta is modified (meta_modified = true).
    /// 2. `start_snapshot` puts us in snapshot mode.
    /// 3. `take_snapshot` captures the task (meta pending).
    /// 4. Task data is modified → the captured meta is frozen into `snapshots` and `data_modified`
    ///    is set as a normal modification.
    /// 5. `SnapshotShardIter::next` yields the pre-encoded item (meta only; data stays live) and
    ///    clears the pending flag.
    #[tokio::test(flavor = "multi_thread")]
    async fn modify_different_category_during_snapshot() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        // Step 1: modify meta only, outside snapshot mode.
        {
            let mut guard = storage.access_mut(task_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Meta, "test");
            assert!(guard.flags.meta_modified());
            assert!(!guard.flags.data_modified());
        }

        // Step 2: enter snapshot mode.
        let (snapshot_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);

        // Step 3: take_snapshot captures the task.
        let shards = storage.take_snapshot(snapshot_guard, &dummy_process, false);

        // Step 4: modify data during snapshot.
        {
            let mut guard = storage.access_mut(task_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Data, "test");
            assert!(guard.flags.data_modified());
            assert!(!guard.flags.meta_modified());
        }

        // Step 5: consume the iterator — must not panic.
        let items: Vec<_> = shards
            .into_iter()
            .flat_map(|shard| shard.into_iter())
            .collect();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].task_id(), task_id);

        {
            let guard = storage.access_mut(task_id);
            // meta was persisted by this snapshot.
            assert!(!guard.flags.meta_modified());
            // data is dirty for the next cycle.
            assert!(guard.flags.data_modified());
            assert!(!guard.flags.any_snapshot_pending());
        }

        // Next snapshot cycle must pick up data_modified.
        let (_guard2, has_modifications) = storage.start_snapshot();
        assert!(
            has_modifications,
            "shard_modified_counts must count the data modification made after the capture"
        );
    }

    /// With `drain_entries = true` (shutdown path), the modified entries are moved out of the map
    /// (during the scan) and serialized by the iterator, freeing each task's memory as it is
    /// persisted rather than retaining it until the whole snapshot is written. Either way the
    /// entry must be gone from the map by the time the snapshot is consumed.
    #[tokio::test(flavor = "multi_thread")]
    async fn drain_entries_removes_entry_from_map() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        // Modify the task outside snapshot mode so it lands in the modified list.
        {
            let mut guard = storage.access_mut(task_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        }
        assert!(storage.map.get(&task_id).is_some());

        let (snapshot_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);

        // Take the snapshot in drain mode.
        let shards = storage.take_snapshot(snapshot_guard, &dummy_process, true);

        // Consume the iterator: the task is serialized and then removed from the map.
        let items: Vec<_> = shards
            .into_iter()
            .flat_map(|shard| shard.into_iter())
            .collect();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].task_id(), task_id);

        // The entry must be gone from the map now that it has been persisted.
        assert!(
            storage.map.get(&task_id).is_none(),
            "task entry should be removed from the map after being persisted in drain mode"
        );
    }

    /// In drain mode, fully consuming the iterators should release each drained shard's table
    /// allocation entirely (reset-to-empty in `SnapshotShardIter::drop`), not just shrink it.
    #[tokio::test(flavor = "multi_thread")]
    async fn drain_entries_releases_drained_shards() {
        // dashmap requires at least 2 shards.
        let storage = Storage::new(2, true, false);

        // Insert and modify enough tasks to grow the shards' tables beyond their minimum.
        let task_ids: Vec<_> = (1..=256).map(non_transient_task).collect();
        for &task_id in &task_ids {
            let mut guard = storage.access_mut(task_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        }
        let grown_capacity: usize = storage
            .map
            .shards()
            .iter()
            .map(|s| s.read().capacity())
            .sum();
        assert!(grown_capacity >= task_ids.len());

        let (snapshot_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);

        let shards = storage.take_snapshot(snapshot_guard, &dummy_process, true);
        let items: Vec<_> = shards
            .into_iter()
            .flat_map(|shard| shard.into_iter())
            .collect();
        assert_eq!(items.len(), task_ids.len());

        // Every shard is now empty and its table allocation has been released (capacity 0),
        // since the reset swaps in the allocation-free default table.
        for shard in storage.map.shards() {
            let shard = shard.read();
            assert_eq!(shard.len(), 0);
            assert_eq!(
                shard.capacity(),
                0,
                "drained shard should have released its table allocation"
            );
        }
    }

    /// In drain mode, `take_snapshot`'s scan removes *both* kinds of entry from the map: unmodified
    /// entries are erased and freed (never serialized), and the remaining modified-only table is
    /// moved out into the shard iterators (to be serialized, then freed as each is consumed). So
    /// the map is already empty when `take_snapshot` returns, and only the modified task is
    /// yielded.
    #[tokio::test(flavor = "multi_thread")]
    async fn drain_entries_removes_unmodified_during_take_snapshot() {
        let storage = Storage::new(2, true, false);
        let modified_id = non_transient_task(1);
        let unmodified_id = non_transient_task(2);

        // One modified task (gets serialized) and one unmodified task (e.g. restored from disk but
        // never dirtied) that just occupies memory and must not be serialized.
        {
            let mut guard = storage.access_mut(modified_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        }
        // `access_mut` inserts an entry; leaving it without track_modification keeps it unmodified.
        let _ = storage.access_mut(unmodified_id);
        assert!(storage.map.get(&unmodified_id).is_some());

        let (snapshot_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);

        let shards = storage.take_snapshot(snapshot_guard, &dummy_process, true);

        // The scan moved the modified table out and freed the unmodified entry, so both ids are
        // already absent from the map before any iterator is consumed.
        assert!(
            storage.map.get(&unmodified_id).is_none(),
            "unmodified entry should be removed during take_snapshot in drain mode"
        );
        assert!(
            storage.map.get(&modified_id).is_none(),
            "modified entry should be moved out of the map during take_snapshot in drain mode"
        );

        // Consuming the iterators yields only the modified task (the unmodified one was never part
        // of the snapshot).
        let items: Vec<_> = shards
            .into_iter()
            .flat_map(|shard| shard.into_iter())
            .collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].task_id(), modified_id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn undo_non_snapshot_reverses_flag_and_counter() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        {
            let mut guard = storage.access_mut(task_id);
            let outcome = guard.track_modification(SpecificTaskDataCategory::Data, "test");
            assert!(guard.flags.data_modified());
            guard.undo_track_modification(outcome);
            assert!(!guard.flags.data_modified());
            assert!(!guard.flags.any_modified());
        }

        // Counter is back to zero: the next snapshot sees no modifications.
        let (_guard, has_modifications) = storage.start_snapshot();
        assert!(
            !has_modifications,
            "undo must decrement the shard counter so no modifications remain"
        );
    }

    /// A second track on an already-modified category returns `NoChange`; undoing it is a no-op and
    /// must NOT clear the real modification recorded by the first track.
    #[tokio::test(flavor = "multi_thread")]
    async fn undo_nochange_preserves_prior_modification() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        let mut guard = storage.access_mut(task_id);
        // First track is the real modification.
        let _first = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        // Second track on the same category changes nothing.
        let second = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        assert!(matches!(second, TrackOutcome::NoChange));
        // Undoing the no-op must leave the prior modification intact.
        guard.undo_track_modification(second);
        assert!(
            guard.flags.data_modified(),
            "undoing a NoChange outcome must not clear a real prior modification"
        );
    }

    /// Undo only reverses the category it tracked: tracking Data then Meta, undoing only the Meta
    /// outcome must leave Data modified and the shard counter still non-zero.
    #[tokio::test(flavor = "multi_thread")]
    async fn undo_only_reverses_its_own_category() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        {
            let mut guard = storage.access_mut(task_id);
            let _data = guard.track_modification(SpecificTaskDataCategory::Data, "test");
            let meta = guard.track_modification(SpecificTaskDataCategory::Meta, "test");
            assert!(guard.flags.meta_modified());
            guard.undo_track_modification(meta);
            assert!(!guard.flags.meta_modified());
            assert!(guard.flags.data_modified());
        }

        // Data is still modified, so the counter is still non-zero.
        let (_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);
    }

    /// A task wholly outside the snapshot (not captured) that is modified during the snapshot is
    /// tracked as a normal modification: `modified` is set and counted, no `snapshots` entry is
    /// created. Undo reverses both.
    #[tokio::test(flavor = "multi_thread")]
    async fn modify_outside_snapshot_during_snapshot_is_normal() {
        let storage = Storage::new(2, true, false);
        let anchor = non_transient_task(1);
        let task_id = non_transient_task(2);
        {
            let mut guard = storage.access_mut(anchor);
            let _ = guard.track_modification(SpecificTaskDataCategory::Meta, "test");
        }
        // Insert the task (unmodified) so it exists in the map.
        let _ = storage.access_mut(task_id);

        let (snapshot_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);
        let shards = storage.take_snapshot(snapshot_guard, &dummy_process, false);
        assert!(storage.snapshot_mode());

        let mut guard = storage.access_mut(task_id);
        let outcome = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        assert!(matches!(
            outcome,
            TrackOutcome::Tracked {
                bumped: true,
                inserted_snapshot: false,
                ..
            }
        ));
        assert!(guard.flags.data_modified());
        assert!(!guard.flags.any_snapshot_pending());
        assert!(!storage.snapshots.contains_key(&task_id));
        assert_eq!(
            storage.shard_modified_counts[storage.shard_index(&task_id)].load(Ordering::Relaxed),
            1
        );

        guard.undo_track_modification(outcome);
        assert!(!guard.flags.data_modified());
        assert_eq!(
            storage.shard_modified_counts[storage.shard_index(&task_id)].load(Ordering::Relaxed),
            0
        );
        drop(guard);
        let items: Vec<_> = shards.into_iter().flatten().collect();
        assert_eq!(items.len(), 1);
    }

    /// If encoding a captured task panics, that task remains in the shard work list so the
    /// shard's drop restores it as modified for the next snapshot.
    #[tokio::test(flavor = "multi_thread")]
    async fn panicking_process_restores_current_captured_task() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);
        {
            let mut guard = storage.access_mut(task_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        }

        let (snapshot_guard, _) = storage.start_snapshot();
        let process = |_: TaskId,
                       _: &TaskStorage,
                       _: SnapshotMask,
                       _: &mut TurboBincodeBuffer|
         -> SnapshotItem { panic!("encoding failed") };
        let shards = storage.take_snapshot(snapshot_guard, &process, false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            shards.into_iter().flatten().collect::<Vec<_>>()
        }));
        assert!(result.is_err());

        let guard = storage.access_mut(task_id);
        assert!(guard.flags.data_modified());
        assert!(!guard.flags.any_snapshot_pending());
        assert_eq!(
            storage.shard_modified_counts[storage.shard_index(&task_id)].load(Ordering::Relaxed),
            1
        );
    }

    /// A captured task tracked again before being persisted stores a pre-mutation encoded item in
    /// `snapshots`. Undo must remove that item and the new `modified` flag/count, while leaving
    /// the captured (pending) state intact.
    #[tokio::test(flavor = "multi_thread")]
    async fn undo_after_copy_on_write_removes_item_preserves_pending() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        // Modify before snapshot so the category is captured.
        {
            let mut guard = storage.access_mut(task_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        }

        let (snapshot_guard, _) = storage.start_snapshot();
        let shards = storage.take_snapshot(snapshot_guard, &dummy_process, false);

        {
            let mut guard = storage.access_mut(task_id);
            let outcome = guard.track_modification(SpecificTaskDataCategory::Data, "test");
            assert!(matches!(
                outcome,
                TrackOutcome::Tracked {
                    bumped: true,
                    inserted_snapshot: true,
                    ..
                }
            ));
            assert!(storage.snapshots.contains_key(&task_id));

            guard.undo_track_modification(outcome);
            assert!(!guard.flags.data_modified());
            assert!(
                guard.flags.data_snapshot_pending(),
                "the captured modification belongs to the snapshot and must survive undo"
            );
            assert!(
                !storage.snapshots.contains_key(&task_id),
                "undo must remove the pre-mutation item it inserted"
            );
        }
        assert_eq!(
            storage.shard_modified_counts[storage.shard_index(&task_id)].load(Ordering::Relaxed),
            0
        );

        // The captured state is still persisted (from the live task, since no entry remains).
        let items: Vec<_> = shards.into_iter().flatten().collect();
        assert_eq!(items.len(), 1);
        assert!(!storage.access_mut(task_id).flags.any_snapshot_pending());
    }

    /// The capture runs inside the operation exclusion: it moves the modified flags to
    /// `*_snapshot_pending`, keeps `new_task` until persistence, and resets the shard counters.
    #[tokio::test(flavor = "multi_thread")]
    async fn take_snapshot_captures_mask_and_clears_live_bits() {
        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);
        {
            let mut guard = storage.access_mut(task_id);
            guard.flags.set_new_task(true);
            let _ = guard.track_modification(SpecificTaskDataCategory::Meta, "test");
        }

        let (snapshot_guard, _) = storage.start_snapshot();
        let process = |id: TaskId,
                       inner: &TaskStorage,
                       mask: SnapshotMask,
                       buffer: &mut TurboBincodeBuffer| {
            assert_eq!(
                mask,
                SnapshotMask {
                    meta: true,
                    data: false,
                    new_task: true
                }
            );
            dummy_process(id, inner, mask, buffer)
        };
        let shards = storage.take_snapshot(snapshot_guard, &process, false);
        {
            let guard = storage.access_mut(task_id);
            assert!(!guard.flags.any_modified());
            assert!(guard.flags.meta_snapshot_pending());
            assert!(!guard.flags.data_snapshot_pending());
            assert!(guard.flags.new_task());
        }
        assert!(
            storage
                .shard_modified_counts
                .iter()
                .all(|c| c.load(Ordering::Relaxed) == 0)
        );

        let items: Vec<_> = shards.into_iter().flatten().collect();
        assert_eq!(items.len(), 1);
        let guard = storage.access_mut(task_id);
        assert!(!guard.flags.any_snapshot_pending());
        assert!(!guard.flags.new_task());
        assert!(!guard.flags.any_modified());
    }

    /// Captured tasks that are never yielded (the shards or their iterators are dropped early,
    /// e.g. because persisting failed) must be marked as modified again so no change is lost.
    #[tokio::test(flavor = "multi_thread")]
    async fn dropping_shards_early_restores_pending_as_modified() {
        let storage = Storage::new(2, true, false);
        let task_ids: Vec<_> = (1..=8).map(non_transient_task).collect();
        for &task_id in &task_ids {
            let mut guard = storage.access_mut(task_id);
            let _ = guard.track_modification(SpecificTaskDataCategory::Data, "test");
        }
        // One task is also frozen into `snapshots` by a modification after the capture.
        let frozen = task_ids[0];

        let (snapshot_guard, _) = storage.start_snapshot();
        let shards = storage.take_snapshot(snapshot_guard, &dummy_process, false);
        {
            let mut guard = storage.access_mut(frozen);
            let _ = guard.track_modification(SpecificTaskDataCategory::Meta, "test");
        }
        assert!(storage.snapshots.contains_key(&frozen));

        // Consume one item from the first shard, then drop everything.
        let mut shards = shards.into_iter();
        let mut first = shards.next().unwrap().into_iter();
        assert!(first.next().is_some());
        drop(first);
        drop(shards);

        assert!(storage.snapshots.is_empty());
        let mut restored = 0;
        for &task_id in &task_ids {
            let guard = storage.access_mut(task_id);
            assert!(!guard.flags.any_snapshot_pending());
            if guard.flags.data_modified() {
                restored += 1;
            }
        }
        // All but the one persisted task have their captured data modification restored.
        assert_eq!(restored, task_ids.len() - 1);
        let (_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);
    }

    /// A task modified before a snapshot and modified again during it must persist the state
    /// as of the first during-snapshot modification. That state is encoded eagerly by
    /// `track_modification`, so later mutations of the live task (including through shared,
    /// interior-mutable cell contents) cannot leak into the persisted bytes, and the snapshot
    /// iterator must yield the pre-encoded item instead of encoding the task again.
    #[tokio::test(flavor = "multi_thread")]
    async fn modify_during_snapshot_persists_pre_encoded_state() {
        fn output(id: u32) -> OutputValue {
            OutputValue::Output(non_transient_task(id))
        }

        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        // Modify the task's meta data outside snapshot mode.
        {
            let mut guard = storage.access_mut(task_id);
            guard.set_output(output(2));
            let _ = guard.track_modification(SpecificTaskDataCategory::Meta, "test");
        }

        // The bytes persistence must write: the meta data as it was before the second mutation.
        let expected = {
            let mut pre = TaskStorage::new();
            pre.set_output(output(2));
            encode_task_contents(
                task_id,
                &pre,
                SpecificTaskDataCategory::Meta,
                &mut TurboBincodeBuffer::new(),
            )
            .unwrap()
        };

        let (snapshot_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);
        let process = |id: TaskId,
                       inner: &TaskStorage,
                       mask: SnapshotMask,
                       buffer: &mut TurboBincodeBuffer| {
            assert_ne!(
                id, task_id,
                "a task with a pre-encoded snapshot item must not be encoded again"
            );
            dummy_process(id, inner, mask, buffer)
        };
        let shards = storage.take_snapshot(snapshot_guard, &process, false);

        // Modify the task again during the snapshot, then keep mutating the live data.
        {
            let mut guard = storage.access_mut(task_id);
            let outcome = guard.track_modification(SpecificTaskDataCategory::Meta, "test");
            assert!(matches!(
                outcome,
                TrackOutcome::Tracked {
                    inserted_snapshot: true,
                    ..
                }
            ));
            guard.set_output(output(3));
        }

        let items: Vec<_> = shards
            .into_iter()
            .flat_map(|shard| shard.into_iter())
            .collect();
        assert_eq!(items.len(), 1);
        let SnapshotItem::Put {
            task_id: item_task_id,
            meta,
            data,
            task_type_hash,
        } = &items[0]
        else {
            panic!("expected a Put item");
        };
        assert_eq!(*item_task_id, task_id);
        assert!(data.is_none(), "data was never modified");
        assert!(task_type_hash.is_none());
        assert_eq!(meta.as_deref(), Some(&expected[..]));

        // Sanity check: the live state really diverged from what was persisted.
        let live = {
            let guard = storage.access_mut(task_id);
            encode_task_contents(
                task_id,
                &guard,
                SpecificTaskDataCategory::Meta,
                &mut TurboBincodeBuffer::new(),
            )
            .unwrap()
        };
        assert_ne!(live, expected);
        assert!(storage.snapshots.get(&task_id).is_none());
    }

    /// Mixed-category race: meta is part of the snapshot, data is not. The first during-snapshot
    /// change hits data, then meta is changed too. The persisted meta must be the pre-snapshot
    /// meta, data must stay live (not persisted in this cycle), and both categories must be dirty
    /// for the next cycle.
    #[tokio::test(flavor = "multi_thread")]
    async fn modify_other_category_then_snapshot_category_during_snapshot() {
        fn output(id: u32) -> OutputValue {
            OutputValue::Output(non_transient_task(id))
        }

        let storage = Storage::new(2, true, false);
        let task_id = non_transient_task(1);

        // Meta is modified before the snapshot; data is clean.
        {
            let mut guard = storage.access_mut(task_id);
            guard.set_output(output(2));
            let _ = guard.track_modification(SpecificTaskDataCategory::Meta, "test");
        }
        let expected_meta = {
            let mut pre = TaskStorage::new();
            pre.set_output(output(2));
            encode_task_contents(
                task_id,
                &pre,
                SpecificTaskDataCategory::Meta,
                &mut TurboBincodeBuffer::new(),
            )
            .unwrap()
        };

        let (snapshot_guard, has_modifications) = storage.start_snapshot();
        assert!(has_modifications);
        let process = |id: TaskId,
                       inner: &TaskStorage,
                       mask: SnapshotMask,
                       buffer: &mut TurboBincodeBuffer| {
            assert_ne!(
                id, task_id,
                "a task with a pre-encoded snapshot item must not be encoded again"
            );
            dummy_process(id, inner, mask, buffer)
        };
        let shards = storage.take_snapshot(snapshot_guard, &process, false);

        // During the snapshot: data first (not part of the snapshot), then meta.
        {
            let mut guard = storage.access_mut(task_id);
            let outcome = guard.track_modification(SpecificTaskDataCategory::Data, "test");
            assert!(matches!(
                outcome,
                TrackOutcome::Tracked {
                    inserted_snapshot: true,
                    ..
                }
            ));
            let outcome = guard.track_modification(SpecificTaskDataCategory::Meta, "test");
            assert!(matches!(
                outcome,
                TrackOutcome::Tracked {
                    inserted_snapshot: false,
                    ..
                }
            ));
            guard.set_output(output(3));
        }

        let items: Vec<_> = shards
            .into_iter()
            .flat_map(|shard| shard.into_iter())
            .collect();
        assert_eq!(items.len(), 1);
        let SnapshotItem::Put { meta, data, .. } = &items[0] else {
            panic!("expected a Put item");
        };
        assert_eq!(meta.as_deref(), Some(&expected_meta[..]));
        assert!(data.is_none(), "data was not part of this snapshot");

        let guard = storage.access_mut(task_id);
        assert!(guard.flags.meta_modified(), "meta change carried forward");
        assert!(guard.flags.data_modified(), "data change carried forward");
        assert!(!guard.flags.any_snapshot_pending());
        drop(guard);
        assert!(storage.snapshots.get(&task_id).is_none());
    }
}
