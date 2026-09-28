# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coordinate asynchronous actor/critic checkpoints on the trainer driver."""

import os


class AsyncCheckpointCoordinator:
    """Publish a PPO checkpoint only after all participating workers finish.

    Keep at most one checkpoint in flight: a subsequent save drains the previous
    one before worker retention callbacks can remove its files. Worker RPCs fan
    out to every rank; only the driver publishes the shared resume tracker.
    """

    def __init__(self, worker_groups, checkpoint_dir):
        self.worker_groups = worker_groups
        self.checkpoint_dir = checkpoint_dir
        self.pending_step = None

    def finalize(self, blocking=False):
        """Progress every async worker group, then publish a fully saved step."""
        if self.pending_step is None:
            return
        complete = True
        for worker_group in self.worker_groups:
            # Do not short-circuit: each role must progress its own collective queue.
            results = worker_group.finalize_async_checkpointing(blocking=blocking)
            complete = bool(results) and all(result is True for result in results) and complete
        if blocking and not complete:
            raise RuntimeError("Blocking checkpoint finalization left pending writes")
        if complete:
            tracker = os.path.join(self.checkpoint_dir, "latest_checkpointed_iteration.txt")
            with open(tracker + ".tmp", "w") as f:
                f.write(str(self.pending_step))
            os.replace(tracker + ".tmp", tracker)
            self.pending_step = None


def prepare_async_checkpoint(trainer):
    """Drain the previous save and return a coordinator if any role saves async."""
    coordinator = getattr(trainer, "_async_checkpoint_coordinator", None)
    if coordinator is None:
        worker_groups = []
        if trainer.config.actor_rollout_ref.actor.get("checkpoint", {}).get("async_save", False):
            worker_groups.append(trainer.actor_rollout_wg)
        if trainer.use_critic and trainer.config.critic.get("checkpoint", {}).get("async_save", False):
            worker_groups.append(trainer.critic_wg)
        if not worker_groups:
            return None
        coordinator = AsyncCheckpointCoordinator(worker_groups, trainer.config.trainer.default_local_dir)
        trainer._async_checkpoint_coordinator = coordinator
    coordinator.finalize(blocking=True)
    return coordinator


def finalize_async_checkpoint(trainer, blocking=False):
    """Progress a trainer's pending checkpoint at aligned training boundaries."""
    coordinator = getattr(trainer, "_async_checkpoint_coordinator", None)
    if coordinator is not None:
        coordinator.finalize(blocking=blocking)
