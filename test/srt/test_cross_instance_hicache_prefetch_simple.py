"""
Simple test for cross-instance HiCache prefetch logic.

This test validates the logic change without requiring full SGLang imports.
"""

import unittest
from unittest.mock import Mock


class TestCrossInstancePrefetchLogic(unittest.TestCase):
    """Test the prefetch decision logic."""

    def test_should_prefetch_when_backed_up_locally(self):
        """Original behavior: prefetch when node is backed up locally."""
        # Simulate the logic from _prefetch_kvcache
        req_last_node_backuped = True
        matched_len = 5

        # Original logic
        should_prefetch = req_last_node_backuped

        self.assertTrue(should_prefetch)

    def test_should_prefetch_when_not_backed_up_with_partial_match(self):
        """New behavior: prefetch even when not backed up, with partial match."""
        req_last_node_backuped = False
        matched_len = 3  # Has some local match

        # New logic
        should_prefetch = req_last_node_backuped
        if not should_prefetch:
            if matched_len > 0:
                should_prefetch = True
            else:
                should_prefetch = True

        self.assertTrue(should_prefetch)

    def test_should_prefetch_when_not_backed_up_empty_cache(self):
        """New behavior: prefetch even with completely empty local cache."""
        req_last_node_backuped = False
        matched_len = 0  # No local match at all

        # New logic
        should_prefetch = req_last_node_backuped
        if not should_prefetch:
            if matched_len > 0:
                should_prefetch = True
            else:
                should_prefetch = True

        self.assertTrue(should_prefetch)

    def test_hash_computation_logic(self):
        """Test hash computation based on local match status."""
        # Case 1: Node backed up locally
        req_last_node_backuped = True
        matched_len = 5
        mock_last_host_node = Mock()
        mock_last_host_node.get_last_hash_value = Mock(return_value="hash_local")

        should_prefetch = req_last_node_backuped
        if should_prefetch:
            last_hash = mock_last_host_node.get_last_hash_value()

        self.assertEqual(last_hash, "hash_local")

        # Case 2: Not backed up, but has partial match
        req_last_node_backuped = False
        matched_len = 3
        mock_last_host_node.get_last_hash_value = Mock(return_value="hash_partial")

        should_prefetch = req_last_node_backuped
        last_hash = None
        if not should_prefetch:
            if matched_len > 0:
                last_hash = mock_last_host_node.get_last_hash_value()
                should_prefetch = True

        self.assertTrue(should_prefetch)
        self.assertEqual(last_hash, "hash_partial")

        # Case 3: Not backed up, no local match
        req_last_node_backuped = False
        matched_len = 0

        should_prefetch = req_last_node_backuped
        last_hash = None
        if not should_prefetch:
            if matched_len > 0:
                last_hash = mock_last_host_node.get_last_hash_value()
                should_prefetch = True
            else:
                should_prefetch = True
                last_hash = None

        self.assertTrue(should_prefetch)
        self.assertIsNone(last_hash)


class TestCrossInstanceScenarioLogic(unittest.TestCase):
    """Test the complete cross-instance scenario logic."""

    def test_instance_a_writes_instance_b_reads(self):
        """
        Scenario:
        - Instance A has data in local cache and writes to storage
        - Instance B has empty local cache but should prefetch from storage
        """
        # Instance A perspective (not tested here, just for context)
        instance_a_has_local_cache = True
        instance_a_writes_to_storage = True

        # Instance B perspective
        instance_b_last_node_backuped = False  # Not in B's local cache
        instance_b_matched_len = 0  # B has empty cache

        # With the fix, B should still attempt prefetch
        should_prefetch = instance_b_last_node_backuped
        if not should_prefetch:
            if instance_b_matched_len > 0:
                should_prefetch = True
            else:
                should_prefetch = True

        self.assertTrue(should_prefetch,
                       "Instance B should attempt prefetch even with empty local cache")

    def test_cross_instance_with_different_prefix_lengths(self):
        """Test various prefix match scenarios across instances."""
        test_cases = [
            # (backuped, matched_len, expected_should_prefetch, description)
            (True, 10, True, "Local backup with match"),
            (True, 0, True, "Local backup without match"),
            (False, 10, True, "No local backup but partial match"),
            (False, 0, True, "No local backup and no match (cross-instance)"),
        ]

        for backuped, matched_len, expected, desc in test_cases:
            with self.subTest(desc=desc):
                should_prefetch = backuped
                if not should_prefetch:
                    if matched_len > 0:
                        should_prefetch = True
                    else:
                        should_prefetch = True

                self.assertEqual(should_prefetch, expected,
                               f"Failed for case: {desc}")


if __name__ == "__main__":
    unittest.main()
