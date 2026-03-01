"""
Test cross-instance HiCache prefetch from shared storage backend.

This test validates that KV cache can be prefetched from a shared storage backend
(e.g., Mooncake Store) even when the requesting SGLang instance has an empty local
RadixCache. This is critical for distributed deployments where multiple instances
share a common storage backend.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import TreeNode


class TestCrossInstanceHiCachePrefetch(unittest.TestCase):
    """Test cross-instance prefetch functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock scheduler with HiCache storage enabled
        self.scheduler = Mock(spec=Scheduler)
        self.scheduler.enable_hicache_storage = True

        # Mock tree cache
        self.tree_cache = Mock(spec=HiRadixCache)
        self.tree_cache.hicache_storage_pass_prefix_keys = False
        self.scheduler.tree_cache = self.tree_cache

        # Mock request
        self.req = Mock()
        self.req.rid = "test_request_1"
        self.req.fill_ids = list(range(100))  # 100 tokens

    def test_prefetch_with_local_backup(self):
        """Test prefetch when last_node is backed up locally (existing behavior)."""
        # Setup: last_node is backed up locally
        self.req.last_node = Mock(spec=TreeNode)
        self.req.last_node.backuped = True
        self.req.last_host_node = Mock(spec=TreeNode)
        self.req.last_host_node.get_last_hash_value = Mock(return_value="hash_abc")
        self.req.last_host_node.get_prefix_hash_values = Mock(return_value=None)
        self.req.prefix_indices = torch.tensor([0, 1, 2, 3, 4])  # 5 tokens matched
        self.req.host_hit_length = 0

        # Mock init_next_round_input
        self.tree_cache.init_next_round_input = Mock()

        # Execute the actual _prefetch_kvcache logic
        from sglang.srt.managers.scheduler import Scheduler

        # Call the method
        Scheduler._prefetch_kvcache(self.scheduler, self.req)

        # Verify prefetch_from_storage was called
        self.tree_cache.prefetch_from_storage.assert_called_once()
        call_args = self.tree_cache.prefetch_from_storage.call_args
        self.assertEqual(call_args[0][0], "test_request_1")  # req_id
        self.assertEqual(call_args[0][2], list(range(5, 100)))  # new_input_tokens
        self.assertEqual(call_args[0][3], "hash_abc")  # last_hash

    def test_prefetch_without_local_backup_with_partial_match(self):
        """Test prefetch when last_node is NOT backed up but has partial local match."""
        # Setup: last_node is NOT backed up locally, but we have some local match
        self.req.last_node = Mock(spec=TreeNode)
        self.req.last_node.backuped = False  # Key difference: not backed up locally
        self.req.last_host_node = Mock(spec=TreeNode)
        self.req.last_host_node.get_last_hash_value = Mock(return_value="hash_xyz")
        self.req.last_host_node.get_prefix_hash_values = Mock(return_value=None)
        self.req.prefix_indices = torch.tensor([0, 1, 2])  # 3 tokens matched locally
        self.req.host_hit_length = 0

        # Mock init_next_round_input
        self.tree_cache.init_next_round_input = Mock()

        # Execute
        from sglang.srt.managers.scheduler import Scheduler

        Scheduler._prefetch_kvcache(self.scheduler, self.req)

        # Verify prefetch_from_storage was called (this is the fix!)
        self.tree_cache.prefetch_from_storage.assert_called_once()
        call_args = self.tree_cache.prefetch_from_storage.call_args
        self.assertEqual(call_args[0][0], "test_request_1")
        self.assertEqual(call_args[0][2], list(range(3, 100)))  # new_input_tokens
        self.assertEqual(call_args[0][3], "hash_xyz")  # last_hash from local match

    def test_prefetch_without_local_backup_empty_cache(self):
        """Test prefetch when requesting instance has completely empty local cache."""
        # Setup: last_node is NOT backed up, and NO local match at all
        self.req.last_node = Mock(spec=TreeNode)
        self.req.last_node.backuped = False
        self.req.last_host_node = Mock(spec=TreeNode)
        self.req.last_host_node.get_last_hash_value = Mock(return_value=None)
        self.req.last_host_node.get_prefix_hash_values = Mock(return_value=None)
        self.req.prefix_indices = torch.tensor([])  # No local match
        self.req.host_hit_length = 0

        # Mock init_next_round_input
        self.tree_cache.init_next_round_input = Mock()

        # Execute
        from sglang.srt.managers.scheduler import Scheduler

        Scheduler._prefetch_kvcache(self.scheduler, self.req)

        # Verify prefetch_from_storage was called with last_hash=None
        self.tree_cache.prefetch_from_storage.assert_called_once()
        call_args = self.tree_cache.prefetch_from_storage.call_args
        self.assertEqual(call_args[0][0], "test_request_1")
        self.assertEqual(call_args[0][2], list(range(0, 100)))  # All tokens
        self.assertIsNone(call_args[0][3])  # last_hash is None (compute from scratch)

    def test_no_prefetch_when_storage_disabled(self):
        """Test that prefetch is skipped when HiCache storage is disabled."""
        self.scheduler.enable_hicache_storage = False
        self.req.last_node = Mock(spec=TreeNode)
        self.req.last_node.backuped = False

        from sglang.srt.managers.scheduler import Scheduler

        Scheduler._prefetch_kvcache(self.scheduler, self.req)

        # Verify prefetch_from_storage was NOT called
        self.tree_cache.prefetch_from_storage.assert_not_called()


class TestCrossInstanceScenario(unittest.TestCase):
    """Integration-style test for cross-instance scenario."""

    def test_cross_instance_workflow(self):
        """
        Simulate the workflow:
        1. Instance A writes KV cache to shared storage (Mooncake)
        2. Instance B (with empty local cache) should be able to prefetch from storage
        """
        # This is a conceptual test - actual integration would require
        # running two SGLang instances with a shared Mooncake backend

        # Instance B perspective: empty local cache, but storage has data
        scheduler_b = Mock(spec=Scheduler)
        scheduler_b.enable_hicache_storage = True

        tree_cache_b = Mock(spec=HiRadixCache)
        tree_cache_b.hicache_storage_pass_prefix_keys = False
        scheduler_b.tree_cache = tree_cache_b

        req_b = Mock()
        req_b.rid = "instance_b_request"
        req_b.fill_ids = list(range(50))
        req_b.last_node = Mock(spec=TreeNode)
        req_b.last_node.backuped = False  # Not in local cache
        req_b.last_host_node = Mock(spec=TreeNode)
        req_b.last_host_node.get_last_hash_value = Mock(return_value=None)
        req_b.last_host_node.get_prefix_hash_values = Mock(return_value=None)
        req_b.prefix_indices = torch.tensor([])  # Empty local cache
        req_b.host_hit_length = 0

        tree_cache_b.init_next_round_input = Mock()

        from sglang.srt.managers.scheduler import Scheduler

        Scheduler._prefetch_kvcache(scheduler_b, req_b)

        # The fix ensures prefetch is attempted even with empty local cache
        tree_cache_b.prefetch_from_storage.assert_called_once()
        call_args = tree_cache_b.prefetch_from_storage.call_args

        # Verify it tries to fetch all tokens from storage
        self.assertEqual(call_args[0][2], list(range(50)))
        # With empty cache, last_hash should be None (compute from scratch)
        self.assertIsNone(call_args[0][3])


if __name__ == "__main__":
    unittest.main()
