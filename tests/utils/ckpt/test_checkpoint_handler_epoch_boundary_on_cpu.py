# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import os
import shutil
import tempfile

import pytest
import torch
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader


class TestAtEpochBoundary:
    """Tests for CheckpointHandler._at_epoch_boundary."""

    def _make_handler(self, monkeypatch, steps_per_epoch, resume_global_step):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        class MockEngine:
            def is_mp_src_rank_with_outputs(self):
                return True

            def get_data_parallel_rank(self):
                return 0

        from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler

        handler = CheckpointHandler(
            engine=MockEngine(),
            train_dataloader=None,
            default_local_dir="/tmp",
            steps_per_epoch=steps_per_epoch,
        )
        handler.resume_global_step = resume_global_step
        return handler

    def test_disabled_when_steps_per_epoch_unknown(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=None, resume_global_step=3183)
        assert handler._at_epoch_boundary() is False

    def test_resume_from_scratch_is_not_boundary(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=0)
        assert handler._at_epoch_boundary() is False

    def test_mid_epoch_step_is_not_boundary(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3683)
        assert handler._at_epoch_boundary() is False

    def test_epoch_boundary_step_detected(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3183)
        assert handler._at_epoch_boundary() is True

    def test_first_epoch_boundary_detected(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=1061)
        assert handler._at_epoch_boundary() is True


class TestLoadDataloaderState:
    """CheckpointHandler loading policy and real SFT dataloader order tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_handler(self, monkeypatch, steps_per_epoch, resume_global_step, train_dataloader=None, dp_rank=0):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        class MockEngine:
            def is_mp_src_rank_with_outputs(self):
                return True

            def get_data_parallel_rank(self):
                return dp_rank

        class MockDataloader:
            def __init__(self):
                self.loaded_state = None

            def load_state_dict(self, state_dict):
                self.loaded_state = state_dict

        from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler

        handler = CheckpointHandler(
            engine=MockEngine(),
            train_dataloader=MockDataloader() if train_dataloader is None else train_dataloader,
            default_local_dir=self.test_dir,
            steps_per_epoch=steps_per_epoch,
        )
        handler.resume_global_step = resume_global_step
        return handler

    def _write_state(self, name="data_0.pt"):
        state = {"_sampler_iter_yielded": 1061, "_num_yielded": 1061}
        torch.save(state, os.path.join(self.test_dir, name))
        return state

    def test_epoch_boundary_state_is_dropped(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3183)
        self._write_state()

        handler._load_dataloader_state(self.test_dir)

        assert handler.train_dataloader.loaded_state is None

    def test_mid_epoch_state_is_restored(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3683)
        state = self._write_state()

        handler._load_dataloader_state(self.test_dir)

        assert handler.train_dataloader.loaded_state == state

    def test_state_restored_when_detection_disabled(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=None, resume_global_step=3183)
        state = self._write_state()

        handler._load_dataloader_state(self.test_dir)

        assert handler.train_dataloader.loaded_state == state

    def test_missing_state_file_keeps_scratch_start(self, monkeypatch):
        handler = self._make_handler(monkeypatch, steps_per_epoch=1061, resume_global_step=3183)

        handler._load_dataloader_state(self.test_dir)

        assert handler.train_dataloader.loaded_state is None

    @pytest.mark.parametrize("num_replicas,dp_rank", [(1, 0), (2, 0), (2, 1)])
    @pytest.mark.parametrize("checkpoint_step", [3, 8])
    def test_sft_resume_preserves_batch_order(self, monkeypatch, num_replicas, dp_rank, checkpoint_step):
        def make_loader():
            dataset = list(range(64))
            sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=dp_rank, shuffle=True, drop_last=True)
            return StatefulDataLoader(
                dataset, batch_size=8 // num_replicas, sampler=sampler, num_workers=0, drop_last=True
            )

        continuous = make_loader()
        expected = []
        for epoch in range(3):
            continuous.sampler.set_epoch(epoch)
            for batch in continuous:
                expected.append(batch.tolist())
                if len(expected) == checkpoint_step:
                    # Save before the for-loop observes StopIteration, as in SFT.
                    torch.save(continuous.state_dict(), os.path.join(self.test_dir, f"data_{dp_rank}.pt"))

        steps_per_epoch = len(continuous)
        assert expected[:steps_per_epoch] != expected[steps_per_epoch : 2 * steps_per_epoch]
        resumed = make_loader()
        handler = self._make_handler(
            monkeypatch,
            steps_per_epoch=steps_per_epoch,
            resume_global_step=checkpoint_step,
            train_dataloader=resumed,
            dp_rank=dp_rank,
        )
        handler._load_dataloader_state(self.test_dir)

        # Both SFT trainers choose the epoch from global_step and call set_epoch
        # before iterating; restoring an exhausted state would skip this epoch.
        actual = expected[:checkpoint_step]
        for epoch in range(checkpoint_step // steps_per_epoch, 3):
            resumed.sampler.set_epoch(epoch)
            actual.extend(batch.tolist() for batch in resumed)

        assert actual == expected
